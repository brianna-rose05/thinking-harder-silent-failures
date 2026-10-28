"""Main analysis. Reads the main-run results, applies hand-review labels, and
prints the tables in the study design (Sections 7 and 8)."""
import csv
import json
import math
import random
from pathlib import Path

HERE = Path(__file__).parent
MODELS = ["sonnet", "gpt", "deepseek"]
NAMES = {"sonnet": "Claude Sonnet 5.5", "gpt": "GPT-6.1 Sol", "deepseek": "DeepSeek V4.1 Flash"}
WAIT_RATE = 33.54  # $ per hour of customer waiting time (Sun et al., 2026)
CAUGHT = {"caught_and_fixed", "caught_and_reported"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def binom_two_sided(b, c):
    """Exact McNemar: two-sided binomial test on discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def fisher_two_sided(a, b, c, d):
    """Fisher's exact test on [[a, b], [c, d]]."""
    n1, n2, m1, n = a + b, c + d, a + c, a + b + c + d
    def p(x):
        return math.comb(n1, x) * math.comb(n2, m1 - x) / math.comb(n, m1)
    p0 = p(a)
    lo, hi = max(0, m1 - n2), min(n1, m1)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= p0 + 1e-12))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = running
    return out


def waiting_seconds(transcript):
    """Seconds the customer waits for the agent: the agent's generation time."""
    d = json.loads(Path(transcript).read_text())
    return sum(m.get("generation_time_seconds") or 0 for m in d["messages"] if m["role"] == "assistant")


def load():
    labels = {}
    lf = HERE / "results" / "review_labels.csv"
    if lf.exists():
        for r in csv.DictReader(open(lf)):
            labels[(r["model"], r["effort"], r["task_id"])] = r
    rows = []
    for m in MODELS:
        f = HERE / "results" / f"main_{m}" / "test" / "results.csv"
        if not f.exists():
            continue
        for r in csv.DictReader(open(f)):
            key = (r["model"], r["effort"], r["task_id"])
            r["draft_outcome"] = r["outcome"]
            if key in labels:
                r["outcome"] = labels[key]["outcome"]
                r["reviewed"] = True
            r["trap_fired"] = r["trap_fired"] == "True"
            r["task_pass"] = r["task_pass"] == "True"
            r["reread"] = r["reread_after_save"] == "True"
            r["caught"] = r["outcome"] in CAUGHT
            r["cost"] = float(r["tok_agent_cost"] or 0)
            r["reasoning"] = int(r["tok_reasoning"] or 0)
            r["wait"] = waiting_seconds(f.parent / "transcripts" / f"{r['model']}_{r['effort']}_{r['task_id']}.json")
            rows.append(r)
    return rows


def main():
    rows = load()
    by = {(r["model"], r["effort"], r["task_id"]): r for r in rows}
    reviewed = sum(1 for r in rows if r.get("reviewed"))
    print(f"{len(rows)} attempts loaded; {reviewed} hand-reviewed\n")

    print("CATCH RATE BY MODEL AND EFFORT (attempts where the trap fired)")
    print(f"{'model':22}{'effort':7}{'fired':>6}{'caught':>8}{'fixed':>7}{'missed':>8}  {'catch rate (95% range)':24}"
          f"{'re-read':>8}{'task pass':>10}{'reason tok':>11}{'cost':>8}{'wait s':>8}")
    summary = {}
    for m in MODELS:
        for e in ["low", "high"]:
            g = [r for r in rows if r["model"] == m and r["effort"] == e]
            f = [r for r in g if r["trap_fired"]]
            k = sum(r["caught"] for r in f)
            fixed = sum(r["outcome"] == "caught_and_fixed" for r in f)
            missed = sum(r["outcome"] == "missed" for r in f)
            lo, hi = wilson(k, len(f))
            rr = sum(r["reread"] for r in f)
            tp = sum(r["task_pass"] for r in g)
            cost = sum(r["cost"] for r in g) / max(1, len(g))
            reas = sum(r["reasoning"] for r in g) / max(1, len(g))
            wait = sum(r["wait"] for r in g) / max(1, len(g))
            summary[(m, e)] = dict(n=len(g), fired=len(f), caught=k, cost=cost, wait=wait)
            rate = f"{k}/{len(f)} = {k / max(1, len(f)):.0%} ({lo:.0%}-{hi:.0%})"
            print(f"{NAMES[m]:22}{e:7}{len(f):>6}{k:>8}{fixed:>7}{missed:>8}  {rate:24}"
                  f"{rr:>8}{tp:>10}{reas:>11.0f}{cost:>8.3f}{wait:>8.1f}")

    print("\nLOW VS HIGH, PAIRED BY TASK (exact McNemar, Holm-corrected across models)")
    ps, lines = [], []
    for m in MODELS:
        b = c = both = neither = 0
        for t in {r["task_id"] for r in rows if r["model"] == m}:
            lo_r, hi_r = by.get((m, "low", t)), by.get((m, "high", t))
            if not lo_r or not hi_r or not (lo_r["trap_fired"] and hi_r["trap_fired"]):
                continue
            if hi_r["caught"] and not lo_r["caught"]:
                b += 1
            elif lo_r["caught"] and not hi_r["caught"]:
                c += 1
            elif lo_r["caught"]:
                both += 1
            else:
                neither += 1
        p = binom_two_sided(b, c)
        ps.append(p)
        lines.append((m, b, c, both, neither, p))
    for (m, b, c, both, neither, p), ph in zip(lines, holm(ps)):
        print(f"{NAMES[m]:22} caught only at high: {b:2}  only at low: {c:2}  both: {both:2}  neither: {neither:2}"
              f"  p = {p:.3f}  Holm p = {ph:.3f}")

    print("\nRE-READING AND CATCHING (pooled, attempts where the trap fired)")
    f = [r for r in rows if r["trap_fired"]]
    a = sum(r["reread"] and r["caught"] for r in f)
    b = sum(r["reread"] and not r["caught"] for r in f)
    c = sum(not r["reread"] and r["caught"] for r in f)
    d = sum(not r["reread"] and not r["caught"] for r in f)
    print(f"re-read & caught {a}, re-read & missed {b}, no re-read & caught {c}, no re-read & missed {d};"
          f" Fisher p = {fisher_two_sided(a, b, c, d):.4f}")
    for m in MODELS:
        for e in ["low", "high"]:
            g = [r for r in f if r["model"] == m and r["effort"] == e]
            print(f"  {NAMES[m]:22}{e:5} re-read after save in {sum(r['reread'] for r in g)}/{len(g)}")

    print("\nCOST (agent tokens at OpenRouter's billed price) AND WAITING TIME")
    for m in MODELS:
        L, H = summary[(m, "low")], summary[(m, "high")]
        extra_catches = H["caught"] - L["caught"]
        extra_cost = (H["cost"] - L["cost"]) * H["n"]
        wait_cost = (H["wait"] - L["wait"]) * H["n"] / 3600 * WAIT_RATE
        per = f"${extra_cost / extra_catches:.2f} per extra catch" if extra_catches > 0 else "no extra catches at high"
        print(f"{NAMES[m]:22} low ${L['cost']:.3f}/attempt, high ${H['cost']:.3f}/attempt "
              f"({H['cost'] / max(L['cost'], 1e-9):.1f}x); wait {L['wait']:.0f}s vs {H['wait']:.0f}s; "
              f"extra waiting valued at ${wait_cost:.2f} over {H['n']} attempts; {per}")

    total = sum(r["cost"] for r in rows)
    print(f"\nTotal agent cost of main runs: ${total:.2f}")

    # one-line summary for the CRM comparison (RQ4)
    print("\nCRM STUDY COMPARISON (failed saves reported as done)")
    crm = {"sonnet": "20 of 20", "gpt": "20 of 20", "deepseek": "18 of 20"}
    for m in MODELS:
        g = [r for r in rows if r["model"] == m and r["trap_fired"]]
        miss = sum(r["outcome"] == "missed" for r in g)
        print(f"{NAMES[m]:22} CRM: {crm[m]};  here: {miss} of {len(g)} missed (both effort levels)")


if __name__ == "__main__":
    main()
