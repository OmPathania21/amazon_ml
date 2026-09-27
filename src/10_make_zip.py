"""Step 10: build the final submission package.

<team>_submission.zip
|-- output/
|   |-- matching_results.tsv
|   `-- candidate_pairs.tsv
|-- code/
|   `-- business_entity_resolution/
|       |-- src/            (all pipeline code)
|       |-- README.md
|       `-- requirements.txt
`-- Documentation_template.md

Usage: python src/10_make_zip.py --team "<team name>"
"""
import argparse
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description="Build the final submission zip")
    p.add_argument("--team", required=True, help="team name (used in the zip file name)")
    p.add_argument("--output-dir", type=Path, default=ROOT / "output")
    p.add_argument("--doc", type=Path, default=ROOT / "Documentation_template.md")
    args = p.parse_args()

    team = re.sub(r"[^A-Za-z0-9_-]+", "_", args.team.strip()) or "team"
    files = {
        "output/matching_results.tsv": args.output_dir / "matching_results.tsv",
        "output/candidate_pairs.tsv": args.output_dir / "candidate_pairs.tsv",
        "code/business_entity_resolution/README.md": ROOT / "README.md",
        "code/business_entity_resolution/requirements.txt": ROOT / "requirements.txt",
        "Documentation_template.md": args.doc,
    }
    for src in sorted((ROOT / "src").glob("*.py")):
        files[f"code/business_entity_resolution/src/{src.name}"] = src

    missing = [str(v) for v in files.values() if not v.is_file()]
    if missing:
        sys.exit("missing files:\n  " + "\n  ".join(missing))

    zip_path = ROOT / f"{team}_submission.zip"
    tmp = zip_path.with_name(zip_path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for arc, src in files.items():
            print(f"   adding {arc}  ({src.stat().st_size / 1e6:.1f} MB)", flush=True)
            z.write(src, arc)
    tmp.replace(zip_path)

    with zipfile.ZipFile(zip_path) as z:
        bad = z.testzip()
        names = z.namelist()
    if bad:
        sys.exit(f"zip is corrupt at {bad}")
    print(f"\n{zip_path}  ({zip_path.stat().st_size / 1e6:.1f} MB, {len(names)} files, integrity OK)")


if __name__ == "__main__":
    main()
