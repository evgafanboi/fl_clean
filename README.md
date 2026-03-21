
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
    - The default public slice ratio is randomized between `5%` to `20%` for each client, with this the public set would be `~14.5%` of the total CICIoT2023 dataset, with the test set being `20%` and the train set being `80% (minus the public slice in some FL cases)`.
    - It also sums the sample count per class for each clien into a table, available in `Markdown` and `Excel` format under the same directory as the partitions.

## Federated Learning

> **see FCIL below**

**Run federated learning simulations:**

```sh
python -m FL --n_clients <> --partition_type <> --strategy <> --rounds <>
```

**Strategies**:
- `FedAvg`[4]: Standard federated averaging
- `FedProx`[5]: Proximal term regularization (`--mu 0.01`)
- `FedDyn`[6]: Dynamic regularization (`--feddyn_alpha 0.1`)
- `RobustFilter`[7]: Byzantine-robust aggregation (`--robust_epsilon` controls the spectral filtering threshold $\epsilon$. Set it higher than the Byzantine ratio)
- `FedCoMed`[9]: Coordinate-wise Median for Byzantine-robust FL
- `DeepFed`[10]: FedAvg using Paillier Homomorphic Encryption (expect extremely long runtime)

**Federated distillation**:
- `Cronus`[8]: Byzantine-robust semi supervised federated distillation, `--robust_epsilon` with value higher than byzantine ratio. `--no_dis` to remove SSFL-IDS's discriminator, which worsen the quality but adhere to the original paper.
- `FedDKD`[11]: Federated Decentralized Knowledge Distillation (deprecated)
- `FedProto`[12]: FedProto `--gamma`, default `1.0`, controls the $\lambda$ in its loss function $\mathcal{L}=\mathcal{L}_S + \lambda\mathcal{L}_R$, where $\mathcal{L}_S$ is defined as the standard supervised loss (cross-entropy in our context) and $\mathcal{L}_R$ is the prototype-distance loss, the distance function is not specific and our code implement L1 distance.
- `FedSSD`[13]: Federated Selective Self Distillation. `--m_max` controls $M_{max}$, as in $M(x)[k_1] M_{max} \cdot [M_{class}[k_1]M_{sample}(x)-0.1]^+$, where $M$ is the distillation weight in the $L_{SSD}$ loss function, $L_{SSD}=\mathbb{E}(||M\odot z^g - M \odot z||_2^2)$. Default `1.0`.
- `SSFL-IDS`[14]: Semi-supervised Federated Learning. `--dis_rounds` controls how many epochs the discriminator is trained, default `3`. `--dist_rounds` controls how many epochs the clients learn the voted public dataset, default `2`.
- `FedMD`[15]: Public dataset distillation.
- `FD`[16]: FederatedDistillation, use `--gamma` to control distillation weight.
- `FLTrust`[17]: Byzantine-robust FL with labeled public dataset (root dataset). `--root_iterations`, default `1`, controls how many iterations the server trains on the root dataset.

**Examples**:
```sh
# FedProx with proximal term
python3 -m FL --n_clients 10 --partition_type label_skew_0.1-10 --strategy FedProx --mu 0.01

# FedDyn with alpha
python3 -m FL --n_clients 10 --partition_type label_skew_0.01-100 --strategy FedDyn --feddyn_alpha 0.1

# FedSSD
python3 -m FL --n_clients 10 --partition_type iid-500 --strategy FedSSD 
```

**General parameters**:
- `--batch_size`: Training batch size (default: 8192)
- `--rounds`: Communication rounds (default: 10)
- `--epochs`: Local epochs per round (default: 5)
- `--model`: default `dense` (MLP), `{gru, dcblstm}` (RNN-GRU, DCNNBiLSTM).


**Outputs**:
- `results/{strategy}_{n_clients}client_{partition}.log`: Training logs
- `results/{strategy}_{n_clients}client_{partition}.xlsx`: Metrics per round

### Poisoning

- To run **label flipping** poisoning, add `--poison label_flip-<ratio>` to the simulation, with `ratio` being `0.1` to `1.0` determining the ratio of clients to be poisoned. 

- Client selection for poisoning are randomized in the first run and the chosen IDs are stored under `results/poison_history` for reproducibility. For example, all simulation runs that use the partition type `label_skew_0.1`, `10 clients` share the same randomized poisoning client selection.

### Checkpointing

- Add per-round checkpoint with `--checkpoint`. Note that the configs are hashed to the checkpoint, so if any hyperparameter change for the same partition type and strategy, or there's no checkpoint at all, the simulation will start fresh (there will be a warning).
- **Important:** Note that checkpointing stores the global model weights and a text holding the run's settings. In the context of _FD strategy_ that does not have a global model, client weights will be stored instead. For large `--n_clients`, checkpoints would consume significant storage. Make sure to clean checkpoints properly. The simulation should cleans up checkpoint automatically if the simulation ended successfully.
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
- **GLFC**[2]: Federated Class Incremental Learning. `--mem` controls exemplar budget $K$, `--encoder_epochs` set the epochs for which the gradient encoder perturbs/reconstruct the prototype samples (default `50`), `--model_selection`: if set, enable best global model selection, server-proxy-client structure, gradient-based communication. Not refined for CICIoT2023 tabular data. If not set, use the most recent global model (default). Experimental argument `--grad_enc small|medium|main` choose from a `7K param`, `23K param` and `main` model for gradient encoder network.

#### For FedSSD

- EWC, LwF, iCaRL, BiC is integrated. Note that FedSSD directly adds another loss term $L_{SSD}$, therefore perturbs the above CIL methods, achieving worse results. FedSSD; however, outperforms FedAvg noticeably in normal FL context (see **Federated Learning** above).

### Citation

[1] Rebuffi, Sylvestre-Alvise, et al. "icarl: Incremental classifier and representation learning." Proceedings of the IEEE conference on Computer Vision and Pattern Recognition. 2017.
[2] Dong, Jiahua, et al. "Federated class-incremental learning." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.
[3] Wang, Fu-Yun, et al. "Foster: Feature boosting and compression for class-incremental learning." European conference on computer vision. Cham: Springer Nature Switzerland, 2022.
[4] McMahan, Brendan, et al. "Communication-efficient learning of deep networks from decentralized data." Artificial intelligence and statistics. Pmlr, 2017.
[5] Li, Tian, et al. "Federated optimization in heterogeneous networks." Proceedings of Machine learning and systems 2 (2020): 429-450.
[6] Acar, Durmus Alp Emre, et al. "Federated learning based on dynamic regularization." arXiv preprint arXiv:2111.04263 (2021).
[7] Chang, Hongyan, et al. "Cronus: Robust and heterogeneous collaborative learning with black-box knowledge transfer." arXiv preprint arXiv:1912.11279 (2019).
[8] Diakonikolas, Ilias, et al. "Being robust (in high dimensions) can be practical." International Conference on Machine Learning. PMLR, 2017.
[9] Yin, Dong, et al. "Byzantine-robust distributed learning: Towards optimal statistical rates." International conference on machine learning. Pmlr, 2018.
[10] Li, Beibei, et al. "DeepFed: Federated deep learning for intrusion detection in industrial cyber–physical systems." IEEE Transactions on Industrial Informatics 17.8 (2020): 5615-5624.
[11] Li, Xinjia, Boyu Chen, and Wenlian Lu. "FedDKD: Federated learning with decentralized knowledge distillation." Applied Intelligence 53.15 (2023): 18547-18563.
[12] Tan, Yue, et al. "Fedproto: Federated prototype learning across heterogeneous clients." Proceedings of the AAAI conference on artificial intelligence. Vol. 36. No. 8. 2022.
[13] He, Yuting, et al. "Learning critically: Selective self-distillation in federated learning on non-iid data." IEEE Transactions on Big Data 10.6 (2022): 789-800.
[14] Zhao, Ruijie, et al. "Semisupervised federated-learning-based intrusion detection method for internet of things." IEEE Internet of Things Journal 10.10 (2022): 8645-8657.
[15] Li, Daliang, and Junpu Wang. "Fedmd: Heterogenous federated learning via model distillation." arXiv preprint arXiv:1910.03581 (2019).
[16] Jeong, Eunjeong, et al. "Communication-efficient on-device machine learning: Federated distillation and augmentation under non-iid private data." arXiv preprint arXiv:1811.11479 (2018).
[17] Cao, Xiaoyu, et al. "Fltrust: Byzantine-robust federated learning via trust bootstrapping." arXiv preprint arXiv:2012.13995 (2020).

#### Classifer based on:
- RNN-GRU: Kasongo, Sydney Mambwe. "A deep learning technique for intrusion detection system using a Recurrent Neural Networks based framework." Computer Communications 199 (2023): 113-125.

- DCNNBiLSTM (dcblstm): Hnamte, Vanlalruata, and Jamal Hussain. "DCNNBiLSTM: An efficient hybrid deep learning-based intrusion detection system." Telematics and Informatics Reports 10 (2023): 100053.

### Acknowledgements

Part of this implementation are adapted from:
- [GLFC](https://github.com/conditionWang/FCIL)
- [FOSTER](https://github.com/G-U-N/ECCV22-FOSTER)

Paper compilation:
- [Awesome-FL](https://youngfish42.github.io/Awesome-FL/)

