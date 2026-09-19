"""Submission CLI.

    python predict.py --input <file|dir> --output acv_predictions.csv [--audit-dir DIR]

Consumes acv_core, so it cannot disagree with the console: both emit the same
`ranked_cars` string the engine returned.  A file that cannot be analysed still
produces a row, because an omitted car scores 0 while a last-placed car
scores 0.125.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import acv_core as core


def collect(target):
    if os.path.isdir(target):
        found = []
        for root, _dirs, files in os.walk(target):
            for name in sorted(files):
                if name.lower().endswith((".xlsx", ".xlsm")) and not name.startswith("~$"):
                    found.append(os.path.join(root, name))
        return sorted(found)
    return [target]


def main(argv=None):
    parser = argparse.ArgumentParser(description="ACV depot ranking -> submission CSV")
    parser.add_argument("--input", required=True, help="workbook or directory of workbooks")
    parser.add_argument("--output", default="acv_predictions.csv", help="submission CSV path")
    parser.add_argument("--audit-dir", default=None, help="write a per-file audit JSON here")
    args = parser.parse_args(argv)

    paths = collect(args.input)
    if not paths:
        print("no .xlsx files found under {}".format(args.input), file=sys.stderr)
        return 2

    results = []
    failures = 0
    for path in paths:
        file_id = os.path.basename(path)
        result = core.analyze_workbook(path, file_id=file_id)
        results.append(result)
        if not result["ok"]:
            failures += 1
            print("WARN {}: {} -- fallback ordering emitted".format(
                file_id, result["error"]), file=sys.stderr)
        print("{:<22} {:<18} {}".format(
            file_id, result["ranking_basis"], result["ranked_cars"]))

        if args.audit_dir:
            os.makedirs(args.audit_dir, exist_ok=True)
            audit = {
                "file_id": file_id,
                "ranked_cars": result["ranked_cars"],
                "ranking_basis": result["ranking_basis"],
                "basis_reason": result["basis_reason"],
                "rows": result["rows"],
                "columns_read": result["columns_read"],
                "columns_total": result["columns_total"],
                "z_margin": result["z_margin"],
                "cars": [
                    {key: info.get(key) for key in (
                        "car", "rank", "status", "shortfall", "samples",
                        "asymmetry", "peak_shortfall", "anomaly", "dropouts",
                        "severity", "filters_applied")}
                    for info in result["cars"]
                ],
                "alerts": [
                    {"kind": a["kind"], "headline": a["headline"], "cars": a["cars"]}
                    for a in result["alerts"]
                ],
                "schema_log": result["schema_log"],
            }
            target = os.path.join(
                args.audit_dir, os.path.splitext(file_id)[0] + ".audit.json")
            with open(target, "w", encoding="utf-8") as handle:
                json.dump(audit, handle, indent=2, default=str)

    with open(args.output, "w", encoding="utf-8", newline="") as handle:
        handle.write(core.submission_csv(results))

    print("\nwrote {} ({} row{}{})".format(
        args.output, len(results), "" if len(results) == 1 else "s",
        ", {} fallback".format(failures) if failures else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
