"""Split attempts into clear misses (no information after the blocked save) and
attempts that need a full read (a re-read, an error, or a repeat save after it)."""
import csv
import json
import sys
from pathlib import Path

WRITES = {"book_reservation", "cancel_reservation", "send_certificate",
          "update_reservation_baggages", "update_reservation_flights",
          "update_reservation_passengers"}
READS = {"get_reservation_details", "get_user_details"}

rows = []
for d in sys.argv[1:]:
    d = Path(d)
    for r in csv.DictReader(open(d / "results.csv")):
        name = f"{r['model']}_{r['effort']}_{r['task_id']}"
        ms = json.loads((d / "transcripts" / f"{name}.json").read_text())["messages"]
        calls = {t["id"]: t for m in ms for t in (m.get("tool_calls") or [])}
        bi = next((i for i, m in enumerate(ms) if m["role"] == "tool" and m["id"] in calls
                   and calls[m["id"]]["name"] in WRITES and not m.get("error")), None)
        if bi is None:
            kind = "trap_not_reached"
        else:
            after = ms[bi + 1:]
            reads = sum(1 for m in after for t in (m.get("tool_calls") or []) if t["name"] in READS)
            errs = sum(1 for m in after if m["role"] == "tool" and m.get("error"))
            saves = sum(1 for m in after for t in (m.get("tool_calls") or []) if t["name"] in WRITES)
            kind = "clear_miss" if reads == 0 and errs == 0 and r["outcome"] == "missed" else "read_in_full"
            if kind == "read_in_full" and reads == 0 and errs == 0 and saves == 0 and r["outcome"] == "missed":
                kind = "clear_miss"
        rows.append((name, r["outcome"], kind, str(d)))
w = csv.writer(sys.stdout)
w.writerow(["name", "draft", "kind", "dir"])
for row in rows:
    w.writerow(row)
