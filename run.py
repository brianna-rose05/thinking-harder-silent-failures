"""Run the study: each task x model x effort level, task by task, resumable.

Stops before any attempt once this study's new OpenRouter spend reaches the
budget (default $30), measured from the key's usage at the first run.
"""
import argparse
import csv
import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
os.chdir(HERE)
sys.path.insert(0, str(HERE))

from loguru import logger  # noqa: E402

logger.remove()
logger.add(sys.stderr, level="WARNING")

from tau2.data_model.simulation import TextRunConfig  # noqa: E402
from tau2.domains.airline.environment import get_tasks  # noqa: E402
from tau2.evaluator.evaluator import EvaluationType  # noqa: E402
from tau2.runner.batch import run_single_task  # noqa: E402

from grade import grade  # noqa: E402
from silent_fault import DOMAIN, register  # noqa: E402

MODELS = {
    "sonnet": "anthropic/claude-sonnet-5.5",
    "gpt": "openai/gpt-6.1-sol",
    "deepseek": "deepseek/deepseek-v4.1-flash-1017",
}
USER_MODEL = "qwen/qwen3.7-flash"  # simulated customer: low cost, not under test
WRITES = {"book_reservation", "cancel_reservation", "send_certificate",
          "update_reservation_baggages", "update_reservation_flights",
          "update_reservation_passengers"}
SEED = 2026


def load_key():
    """OpenRouter key from the environment, or from a .env file next to this script."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    env = HERE / ".env"
    if env.exists():
        for line in open(env):
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("Set OPENROUTER_API_KEY or put it in a .env file next to run.py")


def key_usage(key):
    req = urllib.request.Request("https://openrouter.ai/api/v1/key",
                                 headers={"Authorization": "Bearer " + key})
    return float(json.load(urllib.request.urlopen(req, timeout=30))["data"]["usage"])


def task_sets():
    """20 test tasks and 3 pilot tasks, chosen once with a fixed seed."""
    tasks = [t for t in get_tasks(None)
             if any(a.name in WRITES for a in (t.evaluation_criteria.actions or []))]
    tasks.sort(key=lambda t: int(t.id))
    picked = random.Random(SEED).sample(tasks, 23)
    return picked[:20], picked[20:]


def tokens(messages):
    """Agent token use and exact OpenRouter cost, from the raw API responses."""
    tot = {"prompt": 0, "completion": 0, "reasoning": 0, "cached": 0, "turns": 0, "agent_cost": 0.0}
    for m in messages:
        raw = getattr(m, "raw_data", None)
        if getattr(m, "role", None) != "assistant" or not raw:
            continue
        u = raw.get("usage") or {}
        tot["turns"] += 1
        tot["prompt"] += u.get("prompt_tokens") or 0
        tot["completion"] += u.get("completion_tokens") or 0
        tot["reasoning"] += (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        tot["cached"] += (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        tot["agent_cost"] += u.get("cost") or 0.0
    tot["agent_cost"] = round(tot["agent_cost"], 6)
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["test", "pilot", "smoke"], required=True)
    ap.add_argument("--models", default="sonnet,gpt,deepseek")
    ap.add_argument("--efforts", default="low,high")
    ap.add_argument("--tasks", type=int, default=None, help="only the first N tasks of the set")
    ap.add_argument("--budget", type=float, default=30.0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    key = load_key()
    os.environ["OPENROUTER_API_KEY"] = key
    register()

    test, pilot = task_sets()
    tasks = {"test": test, "pilot": pilot, "smoke": pilot[:2]}[args.set]
    if args.tasks:
        tasks = tasks[: args.tasks]

    out = HERE / args.out / args.set
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    budget_file = HERE / "results" / "budget_start.json"  # one budget across all runs
    if not budget_file.exists():
        budget_file.write_text(json.dumps({"usage_at_start": key_usage(key), "time": time.ctime()}))
    start = json.loads(budget_file.read_text())["usage_at_start"]

    rows_file = out / "results.csv"
    done = set()
    if rows_file.exists():
        done = {(r["model"], r["effort"], r["task_id"]) for r in csv.DictReader(open(rows_file))}

    for task in tasks:
        for mk in args.models.split(","):
            for effort in args.efforts.split(","):
                if (mk, effort, task.id) in done:
                    continue
                before = key_usage(key)
                if before - start >= args.budget - 0.5:
                    print(f"STOP: study spend ${before - start:.2f} reached the ${args.budget} budget")
                    return
                cfg = TextRunConfig(
                    domain=DOMAIN, agent="llm_agent", user="user_simulator",
                    llm_agent="openrouter/" + MODELS[mk],
                    # OpenRouter's own reasoning field: LiteLLM silently drops a
                    # top-level reasoning_effort for these models.
                    llm_args_agent={"extra_body": {"reasoning": {"effort": effort}}},
                    llm_user="openrouter/" + USER_MODEL,
                    # simulator thinking off: it otherwise burns its output on
                    # reasoning and can return an empty reply
                    # low thinking + a high output cap: with thinking off it never
                    # ends the call after a transfer; with no cap it can run out
                    # of output tokens and return an empty reply
                    llm_args_user={"temperature": 0.0, "max_tokens": 8000,
                                   "extra_body": {"reasoning": {"effort": "low"}}},
                    max_steps=100,
                )
                sim, wall = None, 0.0
                for attempt in range(3):
                    t0 = time.time()
                    try:
                        sim = run_single_task(cfg, task, seed=SEED, evaluation_type=EvaluationType.ENV)
                        wall = time.time() - t0
                        break
                    except Exception as e:  # simulator or API hiccup: retry the whole attempt
                        print(f"  retry {attempt + 1} after error: {str(e)[:120]}", flush=True)
                if sim is None:
                    print(f"{task.id:>3} {mk:8} {effort:5} FAILED 3 times, skipped", flush=True)
                    continue
                (out / "transcripts" / f"{mk}_{effort}_{task.id}.json").write_text(sim.model_dump_json(indent=1))
                g = grade(task, sim.messages)
                time.sleep(3)
                spent = key_usage(key) - before
                row = {"model": mk, "effort": effort, "task_id": task.id, **g,
                       **{f"tok_{k}": v for k, v in tokens(sim.messages).items()},
                       "agent_cost_litellm": sim.agent_cost, "user_cost_litellm": sim.user_cost,
                       "openrouter_spend": round(spent, 5), "seconds": round(wall, 1),
                       "termination": str(sim.termination_reason)}
                new = not rows_file.exists()
                fields = ["model", "effort", "task_id", "task_pass", "trap_fired", "outcome",
                          "blocked_tool", "blocked_arguments", "reread_after_save", "redone",
                          "report_phrase_found", "tok_prompt", "tok_completion", "tok_reasoning",
                          "tok_cached", "tok_turns", "tok_agent_cost", "agent_cost_litellm", "user_cost_litellm",
                          "openrouter_spend", "seconds", "termination"]
                with open(rows_file, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                    if new:
                        w.writeheader()
                    w.writerow(row)
                print(f"{task.id:>3} {mk:8} {effort:5} {g['outcome']:22} pass={g['task_pass']} "
                      f"reason_tok={row['tok_reasoning']} ${spent:.3f} {wall:.0f}s", flush=True)


if __name__ == "__main__":
    main()
