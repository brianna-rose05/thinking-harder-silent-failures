"""Write a compact, readable excerpt of each attempt from the blocked save onward,
for hand review. Output: results/review/<model>_<effort>_<task>.txt"""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
WRITES = {"book_reservation", "cancel_reservation", "send_certificate",
          "update_reservation_baggages", "update_reservation_flights",
          "update_reservation_passengers"}


def excerpt(path):
    d = json.loads(Path(path).read_text())
    lines, seen = [], False
    for m in d["messages"]:
        tcs = m.get("tool_calls") or []
        if not seen and any(t["name"] in WRITES for t in tcs):
            seen = True
            lines.append(">>> FIRST SAVE BELOW (the first one that succeeds is the one silently blocked)")
        if not seen:
            continue
        role = m["role"].upper()
        if role == "TOOL":
            lines.append(f"   [tool result{' ERROR' if m.get('error') else ''}] " + (m.get("content") or "")[:220].replace("\n", " "))
            continue
        for t in tcs:
            lines.append(f"{role} CALLS {t['name']} {json.dumps(t['arguments'])[:220]}")
        if m.get("content"):
            lines.append(f"{role}: " + m["content"].replace("\n", " ")[:900])
    return "\n".join(lines)


def main(dirs):
    out = HERE / "results" / "review"
    out.mkdir(parents=True, exist_ok=True)
    for d in dirs:
        rows = list(csv.DictReader(open(Path(d) / "results.csv")))
        for r in rows:
            name = f"{r['model']}_{r['effort']}_{r['task_id']}"
            head = (f"{name} | draft outcome: {r['outcome']} | task pass: {r['task_pass']} | "
                    f"re-read: {r['reread_after_save']} | redone: {r['redone']}\n"
                    f"blocked: {r['blocked_tool']} {r['blocked_arguments']}\n\n")
            (out / f"{name}.txt").write_text(head + excerpt(Path(d) / "transcripts" / f"{name}.json"))


if __name__ == "__main__":
    main(sys.argv[1:])
