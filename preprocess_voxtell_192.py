"""Offline manifest-driven preprocessing for the VoxTell whole-case 192^3 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.voxtell_192 import load_source_manifest, preprocess_source_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess NIfTI cases to offline VoxTell 192^3 artifacts")
    parser.add_argument("--manifest", required=True, help="JSON cases with case_id/image_path/gt_path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--foreground-labels",
        type=int,
        nargs="+",
        required=True,
        help="One or more source GT values treated as binary foreground",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    cases = load_source_manifest(args.manifest)
    rows = [
        preprocess_source_case(case, output_dir / "cases", args.foreground_labels)
        for case in cases
    ]
    output = {
        "format": "voxtell-whole-case-192-v1",
        "source_manifest": str(Path(args.manifest).expanduser().resolve()),
        "foreground_labels": sorted(set(args.foreground_labels)),
        "cases": rows,
    }
    manifest_path = output_dir / "preprocessed_manifest.json"
    manifest_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} offline cases to {manifest_path}")


if __name__ == "__main__":
    main()
