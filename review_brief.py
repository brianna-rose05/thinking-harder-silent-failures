"""Short review view: what the agent did and said after the blocked save."""
import json
import sys
from pathlib import Path

WRITES = {"book_reservation", "cancel_reservation", "send_certificate",
          "update_reservation_baggages", "update_reservation_flights",
          "update_reservation_passengers"}
READS = {"get_reservation_details", "get_user_details"}


def brief(path):
    d = json.loads(Path(path).read_text())
    ms = d["messages"]
    # find blocked save = first successful write
    ids = {}
    for m in ms:
        for t in m.get("tool_calls") or []:
            ids[t["id"]] = t
    blocked_i = None
    for i, m in enumerate(ms):
        if m["role"] == "tool" and m["id"] in ids and ids[m["id"]]["name"] in WRITES and not m.get("error"):
            blocked_i, blocked = i, ids[m["id"]]
            break
    if blocked_i is None:
        return "  (no successful save: trap not reached)"
    out = [f"  BLOCKED {blocked['name']} {json.dumps(blocked['arguments'])[:110]}"]
    for m in ms[blocked_i + 1:]:
        if m["role"] == "assistant":
            for t in m.get("tool_calls") or []:
                tag = "READ" if t["name"] in READS else ("SAVE" if t["name"] in WRITES else "call")
                out.append(f"  {tag} {t['name']} {json.dumps(t['arguments'])[:90]}")
            if m.get("content"):
                out.append("  AGENT: " + m["content"].replace("\n", " ")[:420])
        elif m["role"] == "tool" and m.get("error"):
            out.append("  ERROR: " + (m.get("content") or "")[:140])
        elif m["role"] == "user" and m.get("content"):
            out.append("  CUST: " + m["content"].replace("\n", " ")[:160])
    return "\n".join(out)


for p in sys.argv[1:]:
    print("=" * 8, Path(p).stem)
    print(brief(p))
