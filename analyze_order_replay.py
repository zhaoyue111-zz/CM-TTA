"""Validate and summarize the HEM-last order/replayed-augmentation experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SLIDING_KEYS = (
    "pre_original_sliding",
    "post_original_sliding",
    "pre_selected_sliding",
    "post_selected_sliding",
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError(f"Expected a results.json object with cases: {path}")
    return payload


def _by_case(payload: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    rows = payload["cases"]
    result = {row["basename"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate basenames in {label}")
    return result


def _metrics(row: dict[str, Any]) -> dict[str, float]:
    result = {}
    for key in SLIDING_KEYS:
        result[f"{key}_Dice"] = float(row[key]["Dice"])
    result["final_Precision"] = float(row["Precision"])
    result["final_Recall"] = float(row["Recall"])
    result["response_loss"] = float(row["adaptation_trace"]["response_loss"])
    return result


def _subtract(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {key: left[key] - right[key] for key in left}


def _validate_replay(
    source: dict[str, Any],
    replay: dict[str, Any],
    expected_order: list[str],
    label: str,
) -> None:
    actual_order = [row["basename"] for row in replay["cases"]]
    if actual_order != expected_order:
        raise ValueError(
            f"{label} case order mismatch: expected={expected_order}, actual={actual_order}"
        )
    source_rows = _by_case(source, "source")
    replay_rows = _by_case(replay, label)
    if set(source_rows) != set(replay_rows):
        raise ValueError(f"{label} case set differs from source")
    for basename in expected_order:
        expected = source_rows[basename]["adaptation_trace"]["view_params"]
        actual = replay_rows[basename]["adaptation_trace"]["view_params"]
        if actual != expected:
            raise ValueError(f"{label} view_params mismatch for {basename}")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _print_table(title: str, records: list[dict[str, Any]], value_key: str) -> None:
    print(f"\n{title}")
    print(
        "case\tpre_orig\tpost_orig\tpre_sel\tpost_sel\t"
        "final_precision\tfinal_recall\tresponse_loss"
    )
    for record in records:
        values = record[value_key]
        print(
            f"{record['basename']}\t"
            f"{values['pre_original_sliding_Dice']:.6f}\t"
            f"{values['post_original_sliding_Dice']:.6f}\t"
            f"{values['pre_selected_sliding_Dice']:.6f}\t"
            f"{values['post_selected_sliding_Dice']:.6f}\t"
            f"{values['final_Precision']:.6f}\t"
            f"{values['final_Recall']:.6f}\t"
            f"{values['response_loss']:.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-results",
        type=Path,
        default=Path("results5/9d5_text_response_eps001_w001/results.json"),
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=Path("results5/10order_replay_baseline/results.json"),
    )
    parser.add_argument(
        "--response-results",
        type=Path,
        default=Path("results5/11order_replay_response/results.json"),
    )
    parser.add_argument(
        "--order",
        type=Path,
        default=Path("results5/order_replay_hem_last.json"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("results5/order_replay_comparison.json"),
    )
    args = parser.parse_args()

    source = _load(args.source_results)
    baseline = _load(args.baseline_results)
    response = _load(args.response_results)
    order_payload = json.loads(args.order.read_text(encoding="utf-8"))
    expected_order = order_payload["case_order"]
    _validate_replay(source, baseline, expected_order, "baseline")
    _validate_replay(source, response, expected_order, "response")

    source_rows = _by_case(source, "source")
    baseline_rows = _by_case(baseline, "baseline")
    response_rows = _by_case(response, "response")
    source_order = [row["basename"] for row in source["cases"]]
    records = []
    for basename in expected_order:
        original_response = _metrics(source_rows[basename])
        replay_baseline = _metrics(baseline_rows[basename])
        replay_response = _metrics(response_rows[basename])
        records.append(
            {
                "basename": basename,
                "original_position": source_order.index(basename) + 1,
                "replay_position": expected_order.index(basename) + 1,
                "original_response": original_response,
                "replay_baseline": replay_baseline,
                "replay_response": replay_response,
                "response_minus_same_order_baseline": _subtract(
                    replay_response, replay_baseline
                ),
                "response_new_order_minus_original_order": _subtract(
                    replay_response, original_response
                ),
            }
        )

    hem_names = {"HEM_PA29.nii.gz", "HEM_PA101.nii.gz"}
    hem_records = [row for row in records if row["basename"] in hem_names]
    non_hem_records = [row for row in records if row["basename"] not in hem_names]

    def adaptation_delta(record: dict[str, Any], run: str) -> float:
        values = record[run]
        return (
            values["post_original_sliding_Dice"]
            - values["pre_original_sliding_Dice"]
        )

    summary = {
        "replay_validation_passed": True,
        "actual_replay_order": expected_order,
        "mean_same_order_response_gain_post_original_Dice": _mean(
            [
                row["response_minus_same_order_baseline"][
                    "post_original_sliding_Dice"
                ]
                for row in records
            ]
        ),
        "hem_mean_adaptation_delta_original_order_response": _mean(
            [adaptation_delta(row, "original_response") for row in hem_records]
        ),
        "hem_mean_adaptation_delta_new_order_response": _mean(
            [adaptation_delta(row, "replay_response") for row in hem_records]
        ),
        "non_hem_mean_adaptation_delta_new_order_response": _mean(
            [adaptation_delta(row, "replay_response") for row in non_hem_records]
        ),
    }
    summary["hem_adaptation_delta_change_after_move"] = (
        summary["hem_mean_adaptation_delta_new_order_response"]
        - summary["hem_mean_adaptation_delta_original_order_response"]
    )
    output = {"summary": summary, "cases": records}
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _print_table("Original-order response", records, "original_response")
    _print_table("Reordered baseline", records, "replay_baseline")
    _print_table("Reordered response", records, "replay_response")
    _print_table(
        "Reordered response minus same-order baseline",
        records,
        "response_minus_same_order_baseline",
    )
    print("\nSummary")
    for key, value in summary.items():
        if key != "actual_replay_order":
            print(f"{key}: {value}")
    print(f"comparison_json: {args.json_output}")


if __name__ == "__main__":
    main()
