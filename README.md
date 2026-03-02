
> Used libraries listed in `requirements.txt`.

## Data Preprocessing

##### Extract data/data.zip to CIC23_train.pkl and CIC23_test.pkl

**1. Group class splits** (creates per-class PKL files in `data/CIC23/`):
```sh
# in root dir
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

## Federated Learning

> **see FCIL below**

**Run federated learning simulations:**

```sh
python -m FL --n_clients <> --partition_type <> --strategy <> --rounds <>
```

**Strategies**:
- `FedAvg`: Standard federated averaging
- `FedProx`: Proximal term regularization (`--mu 0.01`)
- `FedDyn`: Dynamic regularization (`--feddyn_alpha 0.1`)
- `RobustFilter`: Byzantine-robust aggregation (ε=0.2)
- `FedCoMed`: Coordinate-wise Median
- `DeepFed`: FedAvg using Paillier Homomorphic Encryption

**Federated distillation**:
- `FedDKD`: Federated Decentralized Knowledge Distillation
- `FedProto`: FedProto
- `FedSSD`: Federated Selective Self Distillation
- `SSFL-IDS`: Semi-supervised Federated Learning
- `FedMD`: FedMD
- `FD`: FederatedDistillation
- `FLTrust`

**Examples**:
```sh
# FedProx with proximal term
python3 -m FL --n_clients 10 --partition_type label_skew-10 --strategy FedProx --mu 0.01

# FedDyn with alpha
python3 -m FL --n_clients 10 --partition_type label_skew-10 --strategy FedDyn --feddyn_alpha 0.1

# FedSSD
python3 -m FL --n_clients 10 --partition_type label_skew-10 --strategy FedSSD
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

### Poisoning

- To run **label flipping** poisoning, add `--poison label_flip-<ratio>` to the simulation, with `ratio` being `0.1` to `1.0` determining the proportion of clients to be poisoned. Selected clients for poisoning are randomized in the first run and their IDs are stored under `results/poison_history` for subsequent re-runs within the same partition type.

---

## Federated Class Incremental Learning

```sh
python -m FCIL --n_clients <> --partition_type <> --strategy <> --rounds <> --cil <CIL method>
```
**Strategies**:
- `FedAvg`: Standard federated averaging
- `FedCoMed`: Coordinate-wise Median
- `FedSSD`: Federated Selective Self Distillation

**Class Incremental Learning method**
#### For FedAvg and FedCoMed:
- **finetune**: lower bound, normal finetune learning without CIL methods.
- **LwF**: learning without forgetting, `--lwf_alpha` (default `0.5`) to control the LwF loss weight, and --lwf_temperature (default `2`) to control distillation temperature.
- **EWC**: Elastic Weight Consolidation, `--ewc_lambda` control EWC $\lambda$, (default `10` - highest performance).
- **iCaRL**[1]: Incremental Classifier and Representation Learning. **LwF** w/ **_exemplars_** management. `--mem` to control memory budget $K$ (default `200`), add `--bce` to use paper's binary cross entropy (default `off`, better for CICIoT23).
    > **note**: `--mem` is used for all CIL methods that use exemplars.
- **BiC**: Large Scale Incremental Learning. Applies linear correction (bias correction) in addition to **iCaRL** exemplar-based learning. `--bic_val_split` to control validation set ratio for each class exemplar set (used to train the linear corrector), `--mem` to control exemplar size (same as **iCaRL**), and all **LwF** arguments `--lwf_alpha` and `--lwf_temperature`.

- **FOSTER**[3]: Feature Boosting and Compression for Class-Incremental Learning. `--beta1` and `--beta2` controls FOSTER effective normalized old/new class numbers $\beta$ used for stage 1 Logits Alignment Loss $L_{LA}$ and stage 2 Compression $L_{BKD}$ (default `0.97`). `--lambda_okd` controls knowledge distillation loss weight (default `1.0`). `--compression_epochs` controls stage 2 compression number of epochs (default `50`).
- **GLFC**[2]: Federated Class Incremental Learning. `--mem` controls exemplar budget $K$, `--encoder_epochs` set the epochs for which the gradient encoder perturbs/reconstruct the prototype samples (default `50`), `--model_selection`: if set, enable best global model selection, server-proxy-client structure, gradient-based communication. Not refined for CICIoT2023 tabular data. If not set, use the most recent global model (default).

#### For FedSSD

- EWC, LwF, iCaRL, BiC is integrated. Note that FedSSD directly adds another loss term $L_{SSD}$, therefore perturbs the above CIL methods, achieving worse results. FedSSD; however, outperforms FedAvg noticeably in normal FL context (see **Federated Learning** above).

### Citation

#### [1] iCaRL (CVPR 2017)
```bibtex
@inproceedings{ rebuffi-cvpr2017,
   author = { Sylvestre-Alvise Rebuffi and Alexander Kolesnikov and Georg Sperl and Christoph H. Lampert },
   title = {{iCaRL:} Incremental Classifier and Representation Learning},
   booktitle = CVPR,
   year = 2017,
}
```

#### [2] GLFC (CVPR 2022)
```bibtex
@InProceedings{dong2022federated,
    author = {Dong, Jiahua and Wang, Lixu and Fang, Zhen and Sun, Gan and Xu, Shichao and Wang, Xiao and Zhu, Qi},
    title = {Federated Class-Incremental Learning},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2022},
}
```

#### [3] FOSTER (ECCV22)

```bibtex
@article{wang2022foster,
  title={FOSTER: Feature Boosting and Compression for Class-Incremental Learning},
  author={Wang, Fu-Yun and Zhou, Da-Wei and Ye, Han-Jia and Zhan, De-Chuan},
  journal={arXiv preprint arXiv:2204.04662},
  year={2022}
}
```

### Acknowledgements

Part of this implementation are adapted from:
- [GLFC](https://github.com/conditionWang/FCIL)
- [FOSTER](https://github.com/G-U-N/ECCV22-FOSTER)
- [FL papers repo](https://github.com/mwan0027/FederatedDistill)

