## Data Preprocessing

##### Extract data/data.zip to CIC23_train.pkl and CIC23_test.pkl

**1. Group class splits** (creates per-class PKL files in `data/CIC23/`):
```sh
cd data/
python CIC23_groupsplit.py
```

- **Note:** The script processes in chunks of 10K samples, edit `line 95` to increase this for faster execution.

**2. Create client partitions**:
```sh
python partition_data.py --num-clients 10 --partition-type iid
```

Partition types:
- `iid`: Equal class distribution per client
- `label_skew`: Dirichlet-based non-IID (α=0.5 default)

Heterogeneity in label skewing: (lower α = more skewed):
```sh
python partition_data.py --num-clients 10 --partition-type label_skew --alpha 0.1
```

- **Note:**
    - The partitioner will further split each partition into a public slice and private slice (train). For simulations that assume a public dataset (auxiliary dataset in `FedSSD`), public slices are concatenated at runtime and clients only use `*_train.npy` for their private dataset. For other simulations, both public and private slices are concatenated at runtime to form a larger set, which will be used for their private dataset. This happens separately between different partitioning runs.
    - The default public slice ratio is `0.285`, with this the public set would be `20%` of the total CICIoT2023 dataset, with the test set being `20%` and the train set being `60%`. If simulations that use public dataset won't be used, best add `--public-ratio 0`.
    - It also sums the sample count per class for each clien into a table, available in `Markdown` and `Excel` format under the same directory as the partitions.

## Training

**Run federated learning simulation**:
```sh
python fedup.py --n_clients 10 --partition_type iid-10 --strategy FedAvg --rounds 10
```

**Strategies**:
- `FedAvg`: Standard federated averaging
- `FedProx`: Proximal term regularization (`--mu 0.01`)
- `FedDyn`: Dynamic regularization (`--feddyn_alpha 0.1`)
- `RobustFilter`: Byzantine-robust aggregation (ε=0.2)

**Examples**:
```sh
# FedProx with proximal term
python fedup.py --n_clients 10 --partition_type label_skew-10 --strategy FedProx --mu 0.01

# FedDyn with alpha
python fedup.py --n_clients 10 --partition_type label_skew-10 --strategy FedDyn --feddyn_alpha 0.1

# RobustFilter
python fedup.py --n_clients 10 --partition_type iid_poisoning-10 --strategy RobustFilter
```

**Parameters**:
- `--batch_size`: Training batch size (default: 8192)
- `--rounds`: Communication rounds (default: 10)
- `--epochs`: Local epochs per round (default: 5)
- `--mu`: FedProx proximal term (default: 0.01)
- `--feddyn_alpha`: FedDyn regularization (default: 0.1, paper optimal)

**Outputs**:
- `results/{strategy}_{n_clients}client_{partition}.log`: Training logs
- `results/{strategy}_{n_clients}client_{partition}.xlsx`: Metrics per round

**Simulations:** `fedmd.py` as **FedMD**, `fed_proto.py` as **FedProto**, `nofed.py` as independent learning (baseline), `fed_distillation.py` as **FederatedDistillation** (Jeong et al. 2018), `feddkd.py` as **FedDKD**, `fedssd.py` as **FedSSD**, `fedup.py` as a united pipeline for **FedAvg**, **FedProx**, **FedDyn**, **FedCoMed**. `ssfl_ids.py` as **SSFL-IDS** (Zhao et al. 2019).

---

Used libraries listed in `requirements.txt`.