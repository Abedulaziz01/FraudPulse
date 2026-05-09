from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline import PipelineConfig, run_project_pipeline


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for the training workflow."""
    parser = argparse.ArgumentParser(description="Run the FraudPulse training pipeline.")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing Fraud_Data.csv, IpAddress_to_Country.csv, and creditcard.csv.",
    )
    parser.add_argument(
        "--artifact-dir",
        default="artifacts",
        help="Directory where trained models, plots, and reports will be written.",
    )
    parser.add_argument(
        "--test-size",
        default=0.2,
        type=float,
        help="Fraction of each dataset reserved for the test split.",
    )
    return parser


def main() -> None:
    """Parse arguments and execute the full project workflow."""
    parser = build_argument_parser()
    args = parser.parse_args()

    config = PipelineConfig(
        data_dir=Path(args.data_dir),
        artifact_dir=Path(args.artifact_dir),
        test_size=args.test_size,
    )
    manifest_path = run_project_pipeline(config)
    print(f"FraudPulse pipeline completed successfully. Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
