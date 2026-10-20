"""Offline preprocessing for the VoxTell whole-case 192^3 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.voxtell_192 import (
    load_source_directory,
    load_source_manifest,
    preprocess_source_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess NIfTI cases to offline VoxTell 192^3 artifacts")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--images-dir",
        help=(
            "Directory containing *.nii.gz images. By default each image is paired "
            "with the same relative filename under the sibling labels directory."
        ),
    )
    source.add_argument(
        "--manifest",
        help="Optional legacy JSON cases with case_id/image_path/gt_path",
    )
    parser.add_argument(
        "--labels-dir",
        help="Override the labels directory used with --images-dir (default: ../labels)",
    )
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
    if args.manifest:
        if args.labels_dir:
            raise ValueError("--labels-dir can only be used with --images-dir")
        cases = load_source_manifest(args.manifest)
        source_metadata = {
            "source_mode": "manifest",
            "source_manifest": str(Path(args.manifest).expanduser().resolve()),
        }
    else:
        cases = load_source_directory(args.images_dir, args.labels_dir)
        images_dir = Path(args.images_dir).expanduser().resolve()
        labels_dir = (
            Path(args.labels_dir).expanduser().resolve()
            if args.labels_dir
            else images_dir.parent / "labels"
        )
        source_metadata = {
            "source_mode": "images_directory",
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir),
        }
    rows = [
        preprocess_source_case(case, output_dir / "cases", args.foreground_labels)
        for case in cases
    ]
    output = {
        "format": "voxtell-whole-case-192-v1",
        **source_metadata,
        "foreground_labels": sorted(set(args.foreground_labels)),
        "cases": rows,
    }
    manifest_path = output_dir / "preprocessed_manifest.json"
    manifest_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} offline cases to {manifest_path}")


if __name__ == "__main__":
    main()
