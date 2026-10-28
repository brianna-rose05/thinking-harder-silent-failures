"""Offline check of the fault layer and grader with scripted agents. No API calls.

For every airline task whose correct answer includes a save:
- a careless agent performs the correct actions once and says it is done
  -> the first save silently fails, the task must FAIL and grade as "missed";
- a careful agent re-reads after its first save, sees nothing changed, and
  saves again -> the task must PASS and grade as "caught_and_fixed";
- the fake response must be identical to what a real save returns.
"""
import sys

from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.domains.airline.environment import get_environment as plain_env
from tau2.domains.airline.environment import get_tasks

from grade import grade
from silent_fault import get_environment as silent_env, register

register()
WRITES = {"book_reservation", "cancel_reservation", "send_certificate",
          "update_reservation_baggages", "update_reservation_flights",
          "update_reservation_passengers"}


def call(env, msgs, n, name, args):
    tc = ToolCall(id=f"c{n}", name=name, arguments=args, requestor="assistant")
    msgs.append(AssistantMessage(role="assistant", content=None, tool_calls=[tc]))
    tm = env.get_response(tc)
    msgs.append(tm)
    return tm


def say(msgs, text):
    msgs.append(AssistantMessage(role="assistant", content=text))


def careless(task):
    env, msgs, n = silent_env(), [], 0
    for a in task.evaluation_criteria.actions:
        n += 1
        call(env, msgs, n, a.name, a.arguments)
    say(msgs, "All done! Your changes have been made.")
    return msgs


def careful(task):
    env, msgs, n, checked = silent_env(), [], 0, False
    for a in task.evaluation_criteria.actions:
        n += 1
        tm = call(env, msgs, n, a.name, a.arguments)
        if a.name in WRITES and not tm.error and not checked:
            checked = True
            rid = a.arguments.get("reservation_id")
            n += 1
            if rid:
                call(env, msgs, n, "get_reservation_details", {"reservation_id": rid})
            else:
                call(env, msgs, n, "get_user_details", {"user_id": a.arguments["user_id"]})
            n += 1
            call(env, msgs, n, a.name, a.arguments)  # save again
    say(msgs, "I re-checked and the first save had not gone through, so I made it again. Done.")
    return msgs


def same_response(task):
    """The blocked save must return exactly what a real save returns."""
    first = next(a for a in task.evaluation_criteria.actions if a.name in WRITES)
    real, fake = plain_env(), silent_env()
    for a in task.evaluation_criteria.actions:
        tc = ToolCall(id="x", name=a.name, arguments=a.arguments, requestor="assistant")
        r, f = real.get_response(tc), fake.get_response(tc)
        if a is first:
            return r.content == f.content and not r.error
        # (no earlier writes before the first one, so states still match)


tasks = [t for t in get_tasks(None)
         if any(a.name in WRITES for a in (t.evaluation_criteria.actions or []))]
bad = []
for t in tasks:
    g1, g2 = grade(t, careless(t)), grade(t, careful(t))
    ok = (g1["outcome"] == "missed" and not g1["task_pass"]
          and g2["outcome"] == "caught_and_fixed" and g2["task_pass"] and g2["reread_after_save"]
          and same_response(t))
    if not ok:
        bad.append((t.id, g1, g2))
print(f"{len(tasks)} tasks with a save; {len(tasks) - len(bad)} pass all checks")
for b in bad:
    print(b)
sys.exit(1 if bad else 0)
