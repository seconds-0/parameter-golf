#!/usr/bin/env python3
"""Generate a Parameter Golf submission folder from experiment results."""

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_DIR / "experiments" / "results"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate a Parameter Golf submission")
    parser.add_argument("--run-id", required=True, help="Run ID from experiments/results/")
    parser.add_argument("--name", required=True, help="Submission name (e.g., TunedMatrixLR)")
    parser.add_argument("--author", default="seconds-0")
    parser.add_argument("--github-id", default="seconds-0")
    parser.add_argument("--track", choices=["10min", "non-record"], default="10min")
    parser.add_argument("--train-script",
                        help="Path to the train_gpt.py used for this run")
    parser.add_argument("--description", default="", help="Approach description for README")
    args = parser.parse_args()

    result_dir = RESULTS_DIR / args.run_id
    if not result_dir.exists():
        print(f"Error: {result_dir} not found", file=sys.stderr)
        sys.exit(1)

    metrics_path = result_dir / "metrics.json"
    if not metrics_path.exists():
        print(f"Error: {metrics_path} not found. Run parse_log.py first.", file=sys.stderr)
        sys.exit(1)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    final = metrics.get("final", {})
    config = metrics.get("config", {})

    if not final.get("val_bpb"):
        print("Error: no final val_bpb in metrics", file=sys.stderr)
        sys.exit(1)

    # Determine output directory
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    track_dir = "track_10min_16mb" if args.track == "10min" else "track_non_record_16mb"
    submission_dir = REPO_DIR / "records" / track_dir / f"{date_str}_{args.name}"

    if submission_dir.exists():
        print(f"Error: {submission_dir} already exists", file=sys.stderr)
        sys.exit(1)

    submission_dir.mkdir(parents=True)

    snapshot_train_script = result_dir / "train_gpt.py"
    if args.train_script:
        train_script_path = Path(args.train_script)
    elif snapshot_train_script.exists():
        train_script_path = snapshot_train_script
    else:
        train_script_path = REPO_DIR / "train_gpt.py"
        print(
            f"Warning: {snapshot_train_script} not found; using current local {train_script_path}",
            file=sys.stderr,
        )

    # 1. submission.json
    code_bytes = train_script_path.stat().st_size if train_script_path.exists() else final.get("code_bytes", 0)

    submission_json = {
        "author": args.author,
        "github_id": args.github_id,
        "name": args.name,
        "blurb": args.description or f"Experiment {args.run_id}",
        "date": datetime.now(timezone.utc).isoformat(),
        "val_loss": final["val_loss"],
        "val_bpb": final["val_bpb"],
        "bytes_total": final.get("total_submission_bytes", 0),
        "bytes_code": code_bytes,
    }
    (submission_dir / "submission.json").write_text(
        json.dumps(submission_json, indent=2) + "\n", encoding="utf-8"
    )

    # 2. train.log
    train_log = result_dir / "train.log"
    if train_log.exists():
        shutil.copy2(train_log, submission_dir / "train.log")
    else:
        print("Warning: train.log not found", file=sys.stderr)

    # 3. train_gpt.py
    if train_script_path.exists():
        shutil.copy2(train_script_path, submission_dir / "train_gpt.py")
    else:
        print(f"Warning: {train_script_path} not found", file=sys.stderr)

    # 4. README.md
    readme = f"""# {args.name}

{args.description or "TODO: Describe your approach here."}

## Configuration
"""
    for k, v in sorted(config.items()):
        readme += f"- `{k}`: {v}\n"

    readme += f"""
## Key Metrics
- **val_bpb**: {final['val_bpb']:.8f} (post-quantization roundtrip)
- **val_loss**: {final['val_loss']:.8f}
- **Total submission size**: {final.get('total_submission_bytes', 'N/A')} bytes
- **Training steps**: {final.get('stop_step', 'N/A')}
- **Peak memory**: {final.get('peak_memory_allocated_mib', 'N/A')} MiB

## Included Files
- `train_gpt.py` — training script snapshot
- `train.log` — full training log
- `submission.json` — leaderboard metadata
"""
    (submission_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"Submission created: {submission_dir}")
    print(f"  val_bpb: {final['val_bpb']:.8f}")
    print(f"  size:    {final.get('total_submission_bytes', 'N/A')} bytes")
    print()
    print("Next steps:")
    print(f"  1. Edit {submission_dir}/README.md with your approach description")
    print(f"  2. git add {submission_dir}")
    print(f"  3. Create a PR to openai/parameter-golf")


if __name__ == "__main__":
    main()
