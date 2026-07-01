"""Command-line entry point.

Usage examples (run from the repo root):

    # End-to-end demo on a generated synthetic slide, no data/model needed:
    python -m pathassist.cli demo --case-id DEMO-001

    # Run on a real image file (any PNG/JPG/TIFF PIL can open):
    python -m pathassist.cli run --case-id CASE-042 --image path/to/slide.png

    # Record a pathologist's decision against a case:
    python -m pathassist.cli review --case-id CASE-042 --decision modify \\
        --reviewer "Dr. Smith" --note "Region 3 is benign."

    # Show the triage worklist (most urgent first) from recorded results:
    python -m pathassist.cli worklist
"""

from __future__ import annotations

import argparse
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
  """Pick the scorer from CLI flags. 'dummy' needs no torch/checkpoint; 'torch'
  loads a trained checkpoint and runs on the chosen device (GPU or CPU)."""
  if args.scorer == "torch":
    if not args.checkpoint:
      raise SystemExit("--checkpoint is required when --scorer torch is used.")
    from .scoring import TorchScorer

    return TorchScorer(args.checkpoint, device_preference=args.device)
  return DummyScorer(seed=args.seed)


def _cmd_demo(args: argparse.Namespace) -> int:
  config = load_config(args.config)
  image = make_synthetic_slide(seed=args.seed)
  store = _build_store(config)
  case, heatmap_path, report_path = analyze_case(
    case_id=args.case_id,
    image=image,
    scorer=_build_scorer(args),
    config=config,
    output_dir=args.output_dir,
    audit_store=store,
  )
  _print_summary(case, heatmap_path, report_path)
  return 0


def _cmd_run(args: argparse.Namespace) -> int:
  config = load_config(args.config)
  image = load_image(args.image)
  store = _build_store(config)
  case, heatmap_path, report_path = analyze_case(
    case_id=args.case_id,
    image=image,
    scorer=_build_scorer(args),
    config=config,
    output_dir=args.output_dir,
    audit_store=store,
  )
  _print_summary(case, heatmap_path, report_path)
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


def _cmd_worklist(args: argparse.Namespace) -> int:
  config = load_config(args.config)
  store = _build_store(config)
  results = store.load_results()
  if not results:
    print("No results recorded yet. Run 'demo' or 'run' first.")
    return 0

  # Keep only the latest record per case, then rank by case score.
  latest: dict[str, dict] = {}
  for record in results:
    result = record["result"]
    latest[result["case_id"]] = result
  ranked = sorted(latest.values(), key=lambda r: r["case_score"], reverse=True)

  print(f"{'PRIORITY':<9} {'SCORE':>6}  {'SUSPICIOUS':>10}  CASE")
  print("-" * 48)
  for result in ranked:
    print(
      f"{result['priority']:<9} {result['case_score']:>6.2f}  "
      f"{result['suspicious_tile_count']:>10}  {result['case_id']}"
    )
  return 0


def _print_summary(case, heatmap_path: Path, report_path: Path) -> None:
  print(f"Case {case.case_id}: priority {case.priority}, score {case.case_score:.2f}")
  print(f"  Suspicious tiles: {case.suspicious_tile_count} / {case.tile_count}")
  print(f"  Heatmap: {heatmap_path}")
  print(f"  Report:  {report_path}")


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    prog="pathassist",
    description="Digital pathology decision-support assistant (scaffold).",
  )
  parser.add_argument("--config", default=None, help="Path to a YAML config file.")
  sub = parser.add_subparsers(dest="command", required=True)

  common_output = {"default": "outputs", "help": "Directory for heatmaps/reports."}

  def _add_scorer_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument(
      "--scorer", default="dummy", choices=["dummy", "torch"],
      help="'dummy' = no-weights heuristic; 'torch' = trained checkpoint.",
    )
    sp.add_argument("--checkpoint", default=None, help="Checkpoint path for --scorer torch.")
    sp.add_argument(
      "--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
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

  p_review = sub.add_parser("review", help="Record a pathologist decision.")
  p_review.add_argument("--case-id", required=True)
  p_review.add_argument(
    "--decision", required=True, choices=ReviewDecision.VALID_DECISIONS
  )
  p_review.add_argument("--reviewer", required=True)
  p_review.add_argument("--note", default="")
  p_review.set_defaults(func=_cmd_review)

  p_worklist = sub.add_parser("worklist", help="Show the ranked triage worklist.")
  p_worklist.set_defaults(func=_cmd_worklist)

  return parser


def main(argv: list[str] | None = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)
  return args.func(args)


if __name__ == "__main__":
  raise SystemExit(main())
