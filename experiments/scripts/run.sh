#!/usr/bin/env bash
# Parameter Golf remote training launcher
# Usage: run.sh <config.yaml> --host <ssh_host> [--gpus N] [--sweep]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$REPO_DIR/experiments/results"

# Parse arguments
CONFIG=""
HOST=""
GPUS=8
SWEEP=false
REMOTE_DIR="/root/code/parameter-golf"

while [[ $# -gt 0 ]]; do
    case $1 in
        --host) HOST="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --sweep) SWEEP=true; shift ;;
        --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
        -*) echo "Unknown option: $1" >&2; exit 1 ;;
        *) CONFIG="$1"; shift ;;
    esac
done

if [[ -z "$CONFIG" ]] || [[ -z "$HOST" ]]; then
    echo "Usage: run.sh <config.yaml> --host <ssh_host> [--gpus N] [--sweep]" >&2
    exit 1
fi

if ! command -v python3 &>/dev/null; then
    echo "Error: python3 not found" >&2
    exit 1
fi

# Parse YAML config into env var assignments
parse_config() {
    local config_file="$1"
    local sweep_index="${2:-}"
    python3 -c "
import yaml, sys, itertools

with open('$config_file') as f:
    cfg = yaml.safe_load(f)

sweep_index = '$sweep_index'

if sweep_index and 'sweep' in cfg:
    base = cfg.get('base_env', cfg.get('env', {}))
    sweep = cfg['sweep']
    params = sweep['params']
    keys = sorted(params.keys())
    combos = list(itertools.product(*(params[k] for k in keys)))
    idx = int(sweep_index)
    if idx >= len(combos):
        print(f'ERROR: sweep index {idx} >= {len(combos)} combinations', file=sys.stderr)
        sys.exit(1)
    combo = dict(zip(keys, combos[idx]))
    env = {**base, **{k: str(v) for k, v in combo.items()}}
    naming = cfg.get('naming', '_'.join(f'{k}{v}' for k, v in combo.items()))
    run_id = naming.format_map({k: str(v) for k, v in combo.items()})
    env['RUN_ID'] = run_id
    print(f'RUN_NAME={cfg[\"name\"]}')
    print(f'SWEEP_TOTAL={len(combos)}')
else:
    env = cfg.get('env', {})
    if 'RUN_ID' not in env:
        import uuid
        env['RUN_ID'] = cfg.get('name', str(uuid.uuid4()))
    print(f'RUN_NAME={cfg.get(\"name\", \"unnamed\")}')

for k, v in env.items():
    print(f'{k}={v}')
"
}

# Get sweep count
get_sweep_count() {
    python3 -c "
import yaml, itertools
with open('$1') as f:
    cfg = yaml.safe_load(f)
if 'sweep' not in cfg:
    print(0)
else:
    params = cfg['sweep']['params']
    keys = sorted(params.keys())
    combos = list(itertools.product(*(params[k] for k in keys)))
    print(len(combos))
"
}

run_single() {
    local config_file="$1"
    local sweep_index="${2:-}"

    # Parse config
    local env_vars
    env_vars=$(parse_config "$config_file" "$sweep_index")
    local run_id
    run_id=$(echo "$env_vars" | grep "^RUN_ID=" | cut -d= -f2)
    local run_name
    run_name=$(echo "$env_vars" | grep "^RUN_NAME=" | cut -d= -f2)

    echo "=== Starting run: $run_id ($run_name) ==="

    # Build env var string for SSH (exclude RUN_NAME and SWEEP_TOTAL)
    local env_str
    env_str=$(echo "$env_vars" | grep -v "^RUN_NAME=" | grep -v "^SWEEP_TOTAL=" | \
        awk -F= '{printf "%s=%s ", $1, $2}')

    # Add data/tokenizer paths
    env_str="$env_str DATA_PATH=$REMOTE_DIR/data/datasets/fineweb10B_sp1024"
    env_str="$env_str TOKENIZER_PATH=$REMOTE_DIR/data/tokenizers/fineweb_1024_bpe.model"

    # Step 1: Sync code
    echo "  Syncing code to $HOST..."
    rsync -az --exclude='.venv' --exclude='data/datasets' --exclude='data/tokenizers' \
        --exclude='experiments/results' --exclude='.codex-reviews' --exclude='.git' \
        --exclude='logs' --exclude='__pycache__' \
        "$REPO_DIR/" "$HOST:$REMOTE_DIR/" 2>/dev/null

    # Step 2: Ensure data
    echo "  Ensuring data is downloaded..."
    ssh "$HOST" "cd $REMOTE_DIR && python3 data/cached_challenge_fineweb.py --variant sp1024 2>&1 | tail -1"

    # Step 3: Install wandb if WANDB_PROJECT is set
    if echo "$env_str" | grep -q "WANDB_PROJECT"; then
        ssh "$HOST" "pip install wandb -q 2>/dev/null" || true
    fi

    # Step 4: Train
    echo "  Launching training ($GPUS GPUs)..."
    ssh "$HOST" "cd $REMOTE_DIR && \
        NCCL_IB_DISABLE=1 \
        $env_str \
        torchrun --standalone --nproc_per_node=$GPUS train_gpt.py 2>&1" | tee /tmp/run_${run_id}_stdout.txt

    # Step 5: Collect results
    echo "  Collecting results..."
    local result_dir="$RESULTS_DIR/$run_id"
    mkdir -p "$result_dir"
    scp "$HOST:$REMOTE_DIR/logs/${run_id}.txt" "$result_dir/train.log" 2>/dev/null || \
        cp /tmp/run_${run_id}_stdout.txt "$result_dir/train.log"
    scp "$HOST:$REMOTE_DIR/final_model.int8.ptz" "$result_dir/" 2>/dev/null || true
    cp "$config_file" "$result_dir/config.yaml"

    # Step 6: Parse log
    echo "  Parsing log..."
    python3 "$SCRIPT_DIR/parse_log.py" "$result_dir/train.log" "$result_dir/metrics.json"

    # Step 7: Summary
    echo ""
    python3 "$SCRIPT_DIR/compare.py" "$result_dir/metrics.json"
    echo ""
    echo "=== Run $run_id complete. Results in $result_dir ==="
}

# Main
if [[ "$SWEEP" == "true" ]]; then
    count=$(get_sweep_count "$CONFIG")
    if [[ "$count" -eq 0 ]]; then
        echo "Error: no sweep config found in $CONFIG" >&2
        exit 1
    fi
    echo "Running sweep with $count configurations on $HOST"
    for i in $(seq 0 $((count - 1))); do
        echo ""
        echo "========== Sweep $((i + 1))/$count =========="
        run_single "$CONFIG" "$i"
    done
    echo ""
    echo "=== Sweep complete. Comparing all runs: ==="
    python3 "$SCRIPT_DIR/compare.py" "$RESULTS_DIR"/*/metrics.json 2>/dev/null || true
else
    run_single "$CONFIG"
fi
