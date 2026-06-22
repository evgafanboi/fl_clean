# FL-IDS Project Guidelines

## Debugging

- Hand test for user in Linux syntax and wait for response. Do not run Python code to Powershell terminal.

## Code Style

Follow `code_rule.instructions.md` (auto-attached):
- No error handling, no try/except, no if-else fail checks
- No comments or docstrings unless on ABC base classes
- Short, dense functions — no one-off helpers
- Python debug/test snippets in Linux bash syntax only

## Run Commands

```sh
# Standard run (PyTorch default)
python -m FL --n_clients 100 --partition_type label_skew_0.1-100 --strategy FedAvg --rounds 20

# SSFL-IDS
python -m FL --n_clients 100 --partition_type label_skew_0.1-100 --strategy SSFL-IDS \
  --dis_rounds 3 --dist_rounds 2 --train_rounds 3 --model gru

# With checkpoint recovery (save every 10 clients)
python -m FL ... --checkpoint 10

# TF backend (required for some legacy architectures)
python -m FL ... --use_tf
```

`python -m FCIL` for class-incremental learning — see [FL/__main__.py](../FL/__main__.py) for all CLI flags.

**Pre-merge partitions before repeated runs** (avoids per-run re-merge overhead):
```sh
python restore_partition.py
```

## Architecture

```
FL/          Standard federated learning simulation
  __main__.py → main.py → pipeline.py   Entry → config → orchestration
  strategy/                             One file per algorithm
  aggregators.py                        Weight-aggregation runtime wrapper
  context.py                            PipelineContext + ClientState
  backend.py                            TF/PT toggle (set once at startup)
  pipeline.py                           Round loop, weight recording, eval
FCIL/        Class-incremental FL (FedAvg/FedCoMed/FedSSD/Centralized only)
models/      TF and PT model definitions (dense, gru, dcblstm, discriminator)
data/        Partition .npy files + global test set
```

**Two pipeline paths** in `main.py`:
- `DISTILLATION_STRATEGIES` → `FDConfig` + `run_distillation_pipeline` (clients keep individual weights)
- Everything else → `FLConfig` + `run_pipeline` (global model aggregated each round)

## Strategies

Weight-aggregation: `FedAvg`, `FedProx`, `FedDyn`, `FedCoMed`, `FLTrust`, `RobustFilter`, `FedMLB`, `DeepFed`, `SecureAggregation`, `NoneStrategy`

Distillation: `FD`, `FedMD`, `FedSSD`, `SSFL-IDS`, `FedProto`, `FedDKD`, `Cronus`, `Ours` (EKD/ABKD — paper's proposed method), `Exp1`

See [FL/strategy/base.py](../FL/strategy/base.py) for the `DistillationStrategy` ABC. Key flags: `has_global_model`, `use_model_pool`, `is_distillation_strategy`.

## Data Layout

```
data/partitions/<N>_client/<partition_type>/
  client_<id>_X_train.npy     private features (80–95% of original)
  client_<id>_y_train.npy
  client_<id>_X_public.npy    public slice (5–20% of original)
  client_<id>_y_public.npy
  client_<id>_X_merged.npy    train+public merged (pre-computed)
data/X_test.npy               global test set
data/y_test.npy
```

`partition_type` CLI format: `<type>_<alpha>-<n_clients>` e.g. `label_skew_0.1-100`. Parsed by `parse_partition_type()` in [FL/data_utils.py](../FL/data_utils.py).

## Backend

Default is **PyTorch**. Pass `--use_tf` for TensorFlow. The flag must be on the CLI — `__main__.py` sets `TF_USE_LEGACY_KERAS=1` before model imports. Check with `FL/backend.py::use_tf()`.

PyTorch build is a **nightly dev wheel** (`torch==2.12.0.dev...+cu128`) — standard `pip install torch` will not match.

## Checkpointing

Distillation strategies support mid-round crash recovery via `FL/strategy/_checkpoint.py`. Payload saved to `checkpoint/<log_stem>/`. Use `--checkpoint N` to save every N clients. Reference implementation: [FL/strategy/Cronus.py](../FL/strategy/Cronus.py).

## Results & Weight Records

| Path | Contents |
|---|---|
| `results/<run>.log` / `.xlsx` | Training logs and metrics |
| `temp_weights/<run>_weight_record/round_<N>/` | `global_weight.bin` (agg) or `client_<id>_weight.bin` (distillation) |
| `checkpoint/<log_stem>/` | Crash-recovery checkpoints |
| `final_results/` | Collected final result files |

Weight records enable post-hoc evaluation replay. `--fresh_run` deletes the weight record directory before starting. `--skip_eval` skips intermediate round evaluation (only evaluates final round).
