"""Command-line entry point.

Usage examples (run from the repo root):

    # End-to-end demo on a generated synthetic slide, no data/model needed:
    python -m pathassist.cli demo --case-id DEMO-001

    # Run on a real image file with the trained ensemble:
    python -m pathassist.cli run --case-id CASE-042 --image path/to/slide.png \\
        --scorer ensemble --checkpoint ensemble.pt

    # Batch-process a folder of images:
    python -m pathassist.cli batch --manifest cases.csv --scorer ensemble \\
        --checkpoint ensemble.pt

    # Record a pathologist's decision against a case:
    python -m pathassist.cli review --case-id CASE-042 --decision modify \\
        --reviewer "Dr. Smith" --note "Region 3 is benign."

    # Export corrections for continuous learning:
    python -m pathassist.cli export-corrections --output corrections.csv

    # Show the triage worklist (most urgent first) from recorded results:
    python -m pathassist.cli worklist
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .audit import AuditStore, ReviewDecision
from .config import load_config
from .pipeline import analyze_case
from .scoring import DummyScorer
from .synthetic import make_synthetic_slide
from .tiling import load_image


def _build_store(config: dict) -> AuditStore:
  return AuditStore(config["audit"]["store_dir"])


def _build_scorer(args: argparse.Namespace):
  """Pick the scorer from CLI flags."""
  checkpoint = args.checkpoint
  if args.organ and not checkpoint:
    from .organs import load_organ_registry, normalize_organ_id

    organ = load_organ_registry().get(normalize_organ_id(args.organ))
    checkpoint = str(organ.checkpoint)
  if args.scorer == "torch":
    if not args.checkpoint:
      raise SystemExit("--checkpoint is required when --scorer torch is used.")
    from .scoring import TorchScorer

    return TorchScorer(args.checkpoint, device_preference=args.device)
  if args.scorer == "ensemble":
    if not args.checkpoint:
      raise SystemExit("--checkpoint is required when --scorer ensemble is used.")
    from .scoring import EnsembleScorer

    return EnsembleScorer(args.checkpoint, device_preference=args.device)
  return DummyScorer(seed=args.seed)


def _cmd_demo(args: argparse.Namespace) -> int:
  config = load_config(args.config)
  image = make_synthetic_slide(seed=args.seed)
  store = _build_store(config)
  case, heatmap_path, report_path, uncertainty_path, _, _ = analyze_case(
    case_id=args.case_id,
    image=image,
    scorer=_build_scorer(args),
    config=config,
    output_dir=args.output_dir,
    audit_store=store,
  )
  _print_summary(case, heatmap_path, report_path, uncertainty_path)
  return 0


def _cmd_run(args: argparse.Namespace) -> int:
  config = load_config(args.config)
  image = load_image(args.image)
  store = _build_store(config)
  case, heatmap_path, report_path, uncertainty_path, _, _ = analyze_case(
    case_id=args.case_id,
    image=image,
    scorer=_build_scorer(args),
    config=config,
    output_dir=args.output_dir,
    audit_store=store,
    image_path=args.image,
  )
  _print_summary(case, heatmap_path, report_path, uncertainty_path)
  return 0


def _load_manifest(path: Path) -> list[tuple[str, Path]]:
  rows: list[tuple[str, Path]] = []
  with path.open("r", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      rows.append((row["case_id"], Path(row["image"])))
  return rows


def _cmd_batch(args: argparse.Namespace) -> int:
  config = load_config(args.config)
  store = _build_store(config)
  scorer = _build_scorer(args)
  manifest = _load_manifest(Path(args.manifest))

  for case_id, image_path in manifest:
    image = load_image(image_path)
    case, heatmap_path, report_path, uncertainty_path, _, _ = analyze_case(
      case_id=case_id,
      image=image,
      scorer=scorer,
      config=config,
      output_dir=args.output_dir,
      audit_store=store,
      image_path=str(image_path),
    )
    _print_summary(case, heatmap_path, report_path, uncertainty_path)

  print(f"Processed {len(manifest)} case(s).")
  return 0


def _cmd_review(args: argparse.Namespace) -> int:
  config = load_config(args.config)
  store = _build_store(config)
  decision = ReviewDecision(
    case_id=args.case_id,
    decision=args.decision,
    reviewer=args.reviewer,
    note=args.note,
  )
  store.record_decision(decision)
  print(f"Recorded '{args.decision}' for case {args.case_id} by {args.reviewer}.")
  return 0


def _cmd_export_corrections(args: argparse.Namespace) -> int:
  config = load_config(args.config)
  store = _build_store(config)
  output_path = store.export_corrections(args.output)
  print(f"Exported corrections to {output_path.resolve()}")
  return 0


def _cmd_worklist(args: argparse.Namespace) -> int:
  config = load_config(args.config)
  store = _build_store(config)
  results = store.load_results()
  if not results:
    print("No results recorded yet. Run 'demo', 'run', or 'batch' first.")
    return 0

  latest: dict[str, dict] = {}
  for record in results:
    result = record["result"]
    latest[result["case_id"]] = result
  ranked = sorted(
    latest.values(),
    key=lambda r: (
      {"URGENT": 0, "REVIEW_QC": 1, "HIGH": 2, "ROUTINE": 3}.get(r["priority"], 99),
      -float(r["case_score"]),
      -float(r.get("max_uncertainty", 0.0)),
    ),
  )

  print(
    f"{'PRIORITY':<10} {'SCORE':>6} {'UNCERT':>7} {'SUSP':>6}  "
    f"{'FLAGS':<24} CASE"
  )
  print("-" * 72)
  for result in ranked:
    flags = ",".join(result.get("review_flags", [])[:2]) or "-"
    print(
      f"{result['priority']:<10} {result['case_score']:>6.2f} "
      f"{result.get('max_uncertainty', 0.0):>7.2f} "
      f"{result['suspicious_tile_count']:>6}  {flags:<24} {result['case_id']}"
    )
  return 0


def _print_summary(case, heatmap_path: Path, report_path: Path, uncertainty_path: Path | None) -> None:
  print(f"Case {case.case_id}: priority {case.priority}, score {case.case_score:.2f}")
  print(f"  Suspicious tiles: {case.suspicious_tile_count} / {case.tile_count}")
  if case.grade:
    print(f"  Grade estimate: {case.grade.grade} ({case.grade.confidence:.2f})")
  if case.review_flags:
    print(f"  Review flags: {', '.join(case.review_flags)}")
  print(f"  Heatmap: {heatmap_path}")
  if uncertainty_path is not None:
    print(f"  Uncertainty map: {uncertainty_path}")
  print(f"  Report:  {report_path}")


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    prog="pathassist",
    description="Digital pathology assistant — triage, explain, report, learn.",
  )
  parser.add_argument("--config", default=None, help="Path to a YAML config file.")
  sub = parser.add_subparsers(dest="command", required=True)

  common_output = {"default": "outputs", "help": "Directory for heatmaps/reports."}

  def _add_scorer_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument(
      "--organ",
      default=None,
      help="Organ id from config/organs.yaml — selects models/<organ>/ensemble.pt.",
    )
    sp.add_argument(
      "--scorer",
      default="dummy",
      choices=["dummy", "torch", "ensemble"],
      help="'dummy' = heuristic; 'torch' = single checkpoint; 'ensemble' = ensemble.pt",
    )
    sp.add_argument(
      "--checkpoint",
      default=None,
      help="Checkpoint path for --scorer torch or ensemble.",
    )
    sp.add_argument(
      "--device",
      default="auto",
      choices=["auto", "cuda", "mps", "cpu"],
      help="Inference device; 'auto' uses GPU if present, else CPU.",
    )

  p_demo = sub.add_parser("demo", help="Run the pipeline on a synthetic slide.")
  p_demo.add_argument("--case-id", default="DEMO-001")
  p_demo.add_argument("--seed", type=int, default=0)
  p_demo.add_argument("--output-dir", **common_output)
  _add_scorer_args(p_demo)
  p_demo.set_defaults(func=_cmd_demo)

  p_run = sub.add_parser("run", help="Run the pipeline on an image file.")
  p_run.add_argument("--case-id", required=True)
  p_run.add_argument("--image", required=True)
  p_run.add_argument("--seed", type=int, default=0)
  p_run.add_argument("--output-dir", **common_output)
  _add_scorer_args(p_run)
  p_run.set_defaults(func=_cmd_run)

  p_batch = sub.add_parser("batch", help="Run the pipeline on many cases from a CSV manifest.")
  p_batch.add_argument(
    "--manifest",
    required=True,
    help="CSV with columns: case_id,image",
  )
  p_batch.add_argument("--output-dir", **common_output)
  _add_scorer_args(p_batch)
  p_batch.set_defaults(func=_cmd_batch)

  p_review = sub.add_parser("review", help="Record a pathologist decision.")
  p_review.add_argument("--case-id", required=True)
  p_review.add_argument(
    "--decision", required=True, choices=ReviewDecision.VALID_DECISIONS
  )
  p_review.add_argument("--reviewer", required=True)
  p_review.add_argument("--note", default="")
  p_review.set_defaults(func=_cmd_review)

  p_export = sub.add_parser(
    "export-corrections",
    help="Export pathologist corrections for retraining.",
  )
  p_export.add_argument("--output", default="corrections.csv")
  p_export.set_defaults(func=_cmd_export_corrections)

  p_worklist = sub.add_parser("worklist", help="Show the ranked triage worklist.")
  p_worklist.set_defaults(func=_cmd_worklist)

  return parser


def main(argv: list[str] | None = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)
  return args.func(args)


if __name__ == "__main__":
  raise SystemExit(main())
