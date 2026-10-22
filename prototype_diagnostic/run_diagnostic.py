from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prototype_diagnostic.build_bank import (
    bank_region_quality_rows,
    build_prototype_bank,
    load_view_params_replay,
    recompute_source_tdc,
    resolve_view_params,
    select_memory_cases,
)
from prototype_diagnostic.evaluate_prototype import (
    evaluate_query_case,
    summarize_query_groups,
)
from prototype_diagnostic.extract_features import build_predictor, embed_prompt
from prototype_diagnostic.utils import (
    case_name,
    foreground_mask,
    load_ras_label_values,
    read_test_entries,
    seed_everything,
    write_csv,
    write_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline leave-one-case-out D5 foreground-prototype diagnostic. "
            "No adaptation, loss, or optimizer is constructed."
        )
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split_file", default=None)
    parser.add_argument("--voxtell_root", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--text_model", required=True)
    parser.add_argument("--prompt", default="liver cancer")
    parser.add_argument(
        "--results_json",
        default=None,
        help=(
            "Optional results.json used only to replay adaptation_trace.view_params; "
            "its TDC/model state is never used for ranking."
        ),
    )
    parser.add_argument("--memory_cases", nargs="+", default=None)
    parser.add_argument("--memory_top_ratio", type=float, default=0.2)
    parser.add_argument("--query_cases", nargs="+", default=None)
    parser.add_argument(
        "--foreground_labels",
        type=int,
        nargs="+",
        default=[1],
        help="One or more integer GT labels counted as foreground during evaluation only.",
    )
    parser.add_argument("--num_views", type=int, default=10)
    parser.add_argument("--view_support", type=float, default=0.8)
    parser.add_argument("--min_region_voxels", type=int, default=64)
    parser.add_argument("--core_erosion", type=int, default=1)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--voxel_chunk_size", type=int, default=262_144)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--text_device", default="cpu")
    parser.add_argument(
        "--text_dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=1377)
    parser.add_argument(
        "--output_dir", default="prototype_diagnostic/outputs/prototype_recovery"
    )
    parser.add_argument(
        "--fail_if_output_exists",
        action="store_true",
        help="Refuse to mix this run with an existing output directory.",
    )
    return parser.parse_args(argv)


def _select_named_entries(entries, requested: list[str] | None):
    by_name = {image.name: (image, label) for image, label in entries}
    if len(by_name) != len(entries):
        raise ValueError("Split contains duplicate image basenames")
    if requested is None:
        return list(by_name)
    names = [case_name(value) for value in requested]
    if len(names) != len(set(names)):
        raise ValueError("Case list contains duplicates")
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError("Unknown query cases: " + ", ".join(unknown))
    return names


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.prompt.strip() != "liver cancer":
        raise ValueError('This diagnostic uses the fixed prompt "liver cancer"')
    if args.num_views < 1 or args.topk < 1 or args.voxel_chunk_size < 1:
        raise ValueError("num_views, topk and voxel_chunk_size must be positive")
    if not args.foreground_labels:
        raise ValueError("--foreground_labels must not be empty")

    output_dir = Path(args.output_dir)
    if args.fail_if_output_exists and output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device)
    text_device = torch.device(args.text_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")

    entries = read_test_entries(args.data_dir, args.split_file)
    entries_by_name = {image.name: (image, label) for image, label in entries}
    query_cases = _select_named_entries(entries, args.query_cases)
    replay_views = load_view_params_replay(args.results_json)
    view_params_source = (
        "results_json_replay_with_deterministic_fallback"
        if args.results_json
        else "deterministic_seed"
    )
    view_params_by_case = resolve_view_params(
        entries_by_name,
        replay_views,
        num_views=args.num_views,
        seed=args.seed,
    )

    predictor = build_predictor(
        args.voxtell_root,
        args.model_dir,
        args.text_model,
        device,
    )
    text_embedding = embed_prompt(
        predictor, args.prompt, text_device, args.text_dtype
    )
    print(
        "[prototype-diagnostic] frozen inference only: "
        f"model_parameters_trainable={sum(p.requires_grad for p in predictor.network.parameters())} "
        f"text_embedding={tuple(text_embedding.shape)}"
    )

    source_tdc_by_case = recompute_source_tdc(
        predictor,
        text_embedding,
        entries_by_name,
        entries_by_name,
        view_params_by_case,
    )
    source_tdc_path = output_dir / "source_tdc.json"
    write_json(
        source_tdc_path,
        {
            "model_state": "frozen source VoxTell",
            "prompt": args.prompt,
            "tdc_type": "sliding_volume_tdc",
            "ranking_metric": "original_tdc",
            "view_params_source": view_params_source,
            "case_order": list(entries_by_name),
            "cases": [source_tdc_by_case[name] for name in entries_by_name],
        },
    )
    memory_cases = select_memory_cases(
        entries_by_name,
        args.memory_cases,
        source_tdc_by_case,
        args.memory_top_ratio,
    )
    config = {
        **vars(args),
        "resolved_memory_cases": memory_cases,
        "resolved_query_cases": query_cases,
        "split_case_order": list(entries_by_name),
        "view_params_source": view_params_source,
        "source_tdc_recomputed": True,
        "tdc_type": "sliding_volume_tdc",
        "memory_ranking_metric": "original_tdc",
        "diagnostic_only": True,
        "uses_ground_truth_for_bank": False,
        "leave_one_case_out": True,
    }
    write_json(output_dir / "config.json", config)

    bank_path = output_dir / "prototype_bank.pt"
    bank = build_prototype_bank(
        predictor,
        text_embedding,
        entries_by_name,
        memory_cases,
        source_tdc_by_case,
        view_params_by_case,
        bank_path,
        num_views=args.num_views,
        view_support=args.view_support,
        min_region_voxels=args.min_region_voxels,
        core_erosion=args.core_erosion,
    )
    bank_gt = {
        name: foreground_mask(
            load_ras_label_values(entries_by_name[name][1]), args.foreground_labels
        )
        for name in memory_cases
    }
    bank_quality_path = output_dir / "bank_region_quality.csv"
    write_csv(bank_quality_path, bank_region_quality_rows(bank, bank_gt))
    prototype_source_cases = {
        str(item["case_name"]) for item in bank["prototypes"]
    }
    no_external_bank = [
        name
        for name in query_cases
        if not any(source != name for source in prototype_source_cases)
    ]
    if no_external_bank:
        raise RuntimeError(
            "Leave-one-case-out would leave no external prototypes for: "
            + ", ".join(no_external_bank)
            + ". Select at least two memory cases that yield retained regions."
        )

    rows = []
    cases_output = output_dir / "cases"
    for index, name in enumerate(query_cases, start=1):
        image_path, label_path = entries_by_name[name]
        metrics = evaluate_query_case(
            predictor,
            text_embedding,
            image_path,
            label_path,
            bank,
            cases_output,
            topk=args.topk,
            voxel_chunk_size=args.voxel_chunk_size,
            foreground_labels=args.foreground_labels,
        )
        rows.append(metrics)
        print(
            f"[prototype-diagnostic] query {index}/{len(query_cases)} {name}: "
            f"pred_fg={metrics['pred_fg_voxels']} "
            f"max_AUROC={metrics['proto_max']['auroc']} "
            f"topk_AUPRC={metrics['proto_topk']['auprc']}"
        )

    write_csv(output_dir / "per_case_metrics.csv", rows)
    summary = summarize_query_groups(rows)
    summary.update(
        {
            "prototype_bank_path": str(bank_path),
            "source_tdc_path": str(source_tdc_path),
            "source_tdc_model_state": "frozen source VoxTell",
            "tdc_type": "sliding_volume_tdc",
            "memory_ranking_metric": "original_tdc",
            "bank_region_quality_path": str(bank_quality_path),
            "prototype_count": len(bank["prototypes"]),
            "memory_case_count": len(memory_cases),
            "query_case_count": len(query_cases),
            "hypothesis": (
                "GT foreground D5 features are more similar than background to "
                "foreground prototypes from other cases"
            ),
        }
    )
    write_json(output_dir / "summary.json", summary)
    print(f"[prototype-diagnostic] results written to {output_dir}")


if __name__ == "__main__":
    main()
