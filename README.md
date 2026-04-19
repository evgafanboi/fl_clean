
> Used libraries listed in `requirements.txt`.

## Data Preprocessing

### Quick data setup for FL (with `CIC23_test.pkl` and `CIC23_train.pkl` already in `data/`) for `500 clients` with **label skewed** Non-IID:
```bash
# split all classes in CIC23_train.pkl to 34 dedicated per-class files
python CIC23_groupsplit.py
# partition data using the splitted 34 files, to 500 clients with label skew setting, Dirichlet alpha 0.1
python partition_data.py --n_clients 500 --partition_type label_skew --alpha 0.1
# since partitions are split to public and private slices (to accomodate public-data FL strategies like FedMD) and FL would take time to merge them into full client partitions for standard FL, pre-create the merged partitions with
python restore_partition.py data/partitions/500_client/label_skew_0.1
```
> After this, FL simulations can be started immediately. See **Federated Learning** section.

##### Extract data/data.zip to CIC23_train.pkl and CIC23_test.pkl

**1. Group class splits** (creates per-class PKL files in `data/CIC23/`):
```sh
# in root dir
python CIC23_groupsplit.py
```

- **Note:** The script processes in chunks of 10K samples, edit `line 95` to increase this for faster execution.

**2. Create client partitions**:
```sh
python partition_data.py --n_clients 10 --partition_type iid
```

Partition types:
- `iid`: Equal class distribution per client
- `label_skew`: Dirichlet-based non-IID (α=0.5 default)

Heterogeneity in label skewing: (lower α = more skewed):
```sh
python partition_data.py --n_clients 10 --partition_type label_skew --alpha 0.1
```

- **Note:**
    - The partitioner will further split each partition into a public slice and private slice (train). For simulations that assume a public dataset (auxiliary dataset in `FedSSD`), public slices are concatenated at runtime and clients only use `*_train.npy` for their private dataset. For other simulations, both public and private slices are concatenated at runtime to form a larger set, which will be used for their private dataset. This happens separately between different partitioning runs.
    - The default public slice ratio is randomized between `5%` to `20%` for each client, with this the public set would be `~14.5%` of the total CICIoT2023 dataset, with the test set being `20%` and the train set being `80% (minus the public slice in some FL cases)`.
    - It also sums the sample count per class for each clien into a table, available in `Markdown` and `Excel` format under the same directory as the partitions.
### IMPORTANT: Pre-merge

Since non-public FL strategies (`FedAvg`, `FedProx`,...) do not use a public dataset, they will attempt to restore the full partitions every run and delete the merged files afterwards. To avoid repeated merging ops and if storage allows, it is always recommended to pre-merge the partitions and have the FL pipeline read them instead of re-merging every time.

```bash
python restore_partition.py data/partitions/<n_clients>/<partition_type>

### for example, for label_skew (alpha=0.1), 100 client partiitons, run:
python restore_partition.py data/partitions/100_client/label_skew_0.1
```


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
- `Ours`: `--kd ekd|abkd`: chooses either `EKD` (**Evidential Knowledge Distillation**[24]) or `ABKD` (**$\alpha-\beta$ Knowledge Distillation**[23]). `--ekd_lambda`: controls `EKD`'s $\lambda$, as in $\mathcal{L}_{EKD} = \mathcal{L}_{1st} + \lambda \mathcal{L}_{2nd}$ (not the weight $\lambda$ used to compute $\alpha$ from $exp(z)$, that value is hard-coded). `--ab_alpha` and `--ab_beta` controls `ABKD`'s $\alpha$ and $\beta$, as in the $\alpha$-$\beta$ divergence. `--kd_epochs` controls the first stage's knowledge distillation epochs, default `2`. Stage 2 infers from the global argument `--epochs` (default `5`). `--ours_temperature` controls `ABKD` distillation temperature, default `4`.
- `Cronus`[8]: Byzantine-robust semi supervised federated distillation, `--robust_epsilon` with value equal or higher than byzantine ratio.
- `FedDKD`[11]: Federated Decentralized Knowledge Distillation (deprecated)
- `FedProto`[12]: FedProto `--gamma`, default `1.0`, controls the $\lambda$ in its loss function $\mathcal{L}=\mathcal{L}_S + \lambda\mathcal{L}_R$, where $\mathcal{L}_S$ is defined as the standard supervised loss (cross-entropy in our context) and $\mathcal{L}_R$ is the prototype-distance loss, the distance function is not specific and our code implement L1 distance.
- `FedSSD`[13]: Federated Selective Self Distillation. `--m_max` controls $M_{max}$, as in $M(x)[k_1] = M_{max} \cdot [M_{class}[k_1]M_{sample}(x)-0.1]^+$, where $M(x)(k_1)$ is the distillation weight for sample $x$ of class $k_1$. In the $L_{SSD}$ loss function, $L_{SSD}=\mathbb{E}(||M\odot z^g - M \odot z||_2^2)$. Default `m_max = 1.0`.
- `SSFL-IDS`[14]: Semi-supervised Federated Learning. `--dis_rounds` controls how many epochs the discriminator is trained, default `3`. `--dist_rounds` controls how many epochs the clients learn the voted public dataset, default `2`.
- `FedMD`[15]: Public dataset distillation.
- `FD`[16]: FederatedDistillation, use `--gamma` to control distillation weight.
- `FLTrust`[17]: Byzantine-robust FL with labeled public dataset (root dataset). `--root_iterations`, default `1`, controls how many iterations the server trains on the root dataset.

> **Note**: Strategies including `FedProto`, `Cronus`, `FD` or `FedMD` does not have a global model, so they perform per-client evaluation on the same test set. To skip evaluating intermediate rounds, add `--skip_eval` which only evaluates the final round's weight records.

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
- `--model`: default `dense` (MLP), `{gru, dcblstm}` (RNN-GRU[18], DCNNBiLSTM[19]).
- **$\triangle$ IMPORTANT**: `--use_tf`: uses **Tensorflow** (for older architecture). Tensorflow was the original backend and will have better stability. If not added, the current default backend **PyTorch** takes place.


**Outputs**:
- `results/{strategy}_{n_clients}client_{partition}.log`: Training logs
- `results/{strategy}_{n_clients}client_{partition}.xlsx`: Metrics per round

### Poisoning

- To run **label flipping** poisoning, add `--poison label_flip-<ratio>` to the simulation, with `ratio` being `0.1` to `1.0` determining the ratio of clients to be poisoned. 

- To run **gradient scaling**, add `--poison gradient_scale <alpha> <ratio>`, where `alpha` is $\alpha$ such that $w_{poisoned} = w_{global} + \alpha \odot w_{local}$. A suggested value for `alpha` is $\alpha = N_{client}$.

- Client selection for poisoning are randomized in the first run and the chosen IDs are stored under `results/poison_history` and reused for reproducibility. For example, all poisoning runs that use the partition setting `label_skew_0.1`, `10 clients` share the same randomized poisoned client selection.

### Checkpointing

- Add client checkpoint with `--checkpoint N`, where N is the the number of clients between checkpoints (for a start, set $N = \frac{N_{clients}}{10}$. Note that the configs are hashed to the checkpoint, so if any hyperparameter change for the same partition type and strategy, or there's no checkpoint at all, the simulation will start fresh (there will be a warning).
- **Important:** Note that checkpointing stores the global model weights and a text holding the run's settings. In the context of _FD strategy_ that does not have a global model, client weights will be stored instead. For large `--n_clients`, checkpoints would consume significant storage. Make sure to clean checkpoints properly. The simulation should cleans up checkpoint automatically if the simulation ended successfully.
---

## Federated Class Incremental Learning

### 1. Further create per-task partitions for clients.

- On existing FL partitions, random class-task mapping and split each client's partitions (both public and private slice) into tasks with `partition_task.py`. Note that the randomized class-task mapping is written to `results/` and used by `FCIL` to determine the class mask to be used at each task.

```bash
python partition_task.py --task_size 5 --benign_class 0 --partition_root data/partitions/ --partition_type label_skew_0.1 --n_clients 500 --data_dir data/CIC23
```
- `--task_size`: default `5`, determines the number of tasks to be created, where each task contains $\frac{N_{class}}{N_{task}}$ classes.
- `--benign_class`: default `0`, determines the class to be always in the first task, for IDS datasets. Should be harmless for other types of dataset.
- `--partition_root`, `--partition_type` and `--n_clients` are used to trace the exact FL partition folder.
- `--data_dir` is used to determine the number of classes, given that there are already per-class files splitted from the training set.

### 2. Run Federated class-incremental learning simulations

```sh
python -m FCIL --n_clients <> --partition_type <> --strategy <> --rounds <> --cil <CIL method>
```
**Strategies**:
- `FedAvg`: Supported for all `CIL` methods.
- `FedCoMed`: Coordinate-wise Median.
- `FedSSD`: Federated Selective Self Distillation. Partially supported.

**Class Incremental Learning method**
#### For FedAvg and FedCoMed:
- **finetune**: lower bound, normal finetune learning without CIL methods.
- **LwF**: learning without forgetting, `--lwf_alpha` (default `0.5`) to control the LwF loss weight, and --lwf_temperature (default `2`) to control distillation temperature.
- **EWC**: Elastic Weight Consolidation, `--ewc_lambda` control EWC $\lambda$, (default `10` - highest tested performance on MLP model).
- **iCaRL**[1]: Incremental Classifier and Representation Learning. **LwF** w/ **_exemplars_** management. `--mem` to control memory budget $K$ (default `200`), add `--bce` to use paper's binary cross entropy (default `off` for CICIoT23).
    > **note**: `--mem` is used for all CIL methods that use exemplars.
- **BiC**[20]: Large Scale Incremental Learning. Applies linear correction (bias correction) in addition to **iCaRL** exemplar-based learning. `--bic_val_split` to control validation set ratio for each class exemplar set (used to train the linear corrector), `--mem` to control exemplar size (same as **iCaRL**), and all **LwF** arguments `--lwf_alpha` and `--lwf_temperature`.

- **FOSTER**[3]: Feature Boosting and Compression for Class-Incremental Learning. `--beta1` and `--beta2` controls FOSTER effective normalized old/new class numbers $\beta$ used for stage 1 Logits Alignment Loss $L_{LA}$ and stage 2 Compression $L_{BKD}$ (default `0.97`). `--lambda_okd` controls knowledge distillation loss weight (default `1.0`). `--compression_epochs` controls stage 2 compression number of epochs (default `50`).
- **GLFC**[2]: Federated Class Incremental Learning. `--mem` controls exemplar budget $K$, `--encoder_epochs` set the epochs for which the gradient encoder perturbs/reconstruct the prototype samples (default `50`), `--model_selection`: if set (adhere to paper), enable best global model selection, server-proxy-client structure, gradient-based communication. Model selection is not refined yet (involve iDLG label reconstruction, gaussian perturbation and gaussian sampling just to choose the best per-round model) and it's not recommended to add it. If so, the most recent global model is used (default for all other CIL methods). Experimental argument `--grad_enc small|medium|main` choose from a `7K param`, `23K param` and `main` model for gradient encoder network.
- **MAS**[21]: Memory Aware Synapses: unsupervised EWC. `--mas_lambda` controls MAS's $\lambda$, default `1.0`
- **PASS**[22]: Exemplar-free, prototype-based CIL. `--pass_gamma` controls $\gamma$ in PASS's loss function $L_{t,total}=L_{t,ce} + \lambda * L_{t,protoAug} + \gamma * L_{t,kd}$. $\lambda$ is hardcoded `1.0`. Since PASS use variance-based prototype sampling, `--proto_size` controls the per-class sample size, default `50`.

#### For FedSSD

- EWC, LwF, iCaRL, BiC is integrated. Note that FedSSD directly adds another loss term $L_{SSD}$, therefore perturbs the above CIL methods, achieving worse results. FedSSD; however, outperforms FedAvg noticeably in normal FL context (see **Federated Learning** above).

### Citation

- [1] Rebuffi, Sylvestre-Alvise, et al. "icarl: Incremental classifier and representation learning." Proceedings of the IEEE conference on Computer Vision and Pattern Recognition. 2017.
- [2] Dong, Jiahua, et al. "Federated class-incremental learning." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.
- [3] Wang, Fu-Yun, et al. "Foster: Feature boosting and compression for class-incremental learning." European conference on computer vision. Cham: Springer Nature Switzerland, 2022.
- [4] McMahan, Brendan, et al. "Communication-efficient learning of deep networks from decentralized data." Artificial intelligence and statistics. Pmlr, 2017.
- [5] Li, Tian, et al. "Federated optimization in heterogeneous networks." Proceedings of Machine learning and systems 2 (2020): 429-450.
- [6] Acar, Durmus Alp Emre, et al. "Federated learning based on dynamic regularization." arXiv preprint arXiv:2111.04263 (2021).
- [7] Chang, Hongyan, et al. "Cronus: Robust and heterogeneous collaborative learning with black-box knowledge transfer." arXiv preprint arXiv:1912.11279 (2019).
- [8] Diakonikolas, Ilias, et al. "Being robust (in high dimensions) can be practical." International Conference on Machine Learning. PMLR, 2017.
- [9] Yin, Dong, et al. "Byzantine-robust distributed learning: Towards optimal statistical rates." International conference on machine learning. Pmlr, 2018.
- [10] Li, Beibei, et al. "DeepFed: Federated deep learning for intrusion detection in industrial cyber–physical systems." IEEE Transactions on Industrial Informatics 17.8 (2020): 5615-5624.
- [11] Li, Xinjia, Boyu Chen, and Wenlian Lu. "FedDKD: Federated learning with decentralized knowledge distillation." Applied Intelligence 53.15 (2023): 18547-18563.
- [12] Tan, Yue, et al. "Fedproto: Federated prototype learning across heterogeneous clients." Proceedings of the AAAI conference on artificial intelligence. Vol. 36. No. 8. 2022.
- [13] He, Yuting, et al. "Learning critically: Selective self-distillation in federated learning on non-iid data." IEEE Transactions on Big Data 10.6 (2022): 789-800.
- [14] Zhao, Ruijie, et al. "Semisupervised federated-learning-based intrusion detection method for internet of things." IEEE Internet of Things Journal 10.10 (2022): 8645-8657.
- [15] Li, Daliang, and Junpu Wang. "Fedmd: Heterogenous federated learning via model distillation." arXiv preprint arXiv:1910.03581 (2019).
- [16] Jeong, Eunjeong, et al. "Communication-efficient on-device machine learning: Federated distillation and augmentation under non-iid private data." arXiv preprint arXiv:1811.11479 (2018).
- [17] Cao, Xiaoyu, et al. "Fltrust: Byzantine-robust federated learning via trust bootstrapping." arXiv preprint arXiv:2012.13995 (2020).
- [20] Wu, Yue, et al. "Large scale incremental learning." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.
- [21] Aljundi, Rahaf, et al. "Memory aware synapses: Learning what (not) to forget." Proceedings of the European conference on computer vision (ECCV). 2018.
- [22] Zhu, Fei, et al. "Prototype augmentation and self-supervision for incremental learning." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.
- [23] Wang, Guanghui, et al. "ABKD: Pursuing a proper allocation of the probability mass in knowledge distillation via $\alpha $-$\beta $-divergence." arXiv preprint arXiv:2505.04560 (2025).
- [24] Xiang, Liangyu, Junyu Gao, and Changsheng Xu. "Evidential Knowledge Distillation." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025.
Xiang, L., Gao, J., & Xu, C. (2025). Evidential Knowledg
#### Classifer based on:
- [18] Kasongo, Sydney Mambwe. "A deep learning technique for intrusion detection system using a Recurrent Neural Networks based framework." Computer Communications 199 (2023): 113-125.

- [19] Hnamte, Vanlalruata, and Jamal Hussain. "DCNNBiLSTM: An efficient hybrid deep learning-based intrusion detection system." Telematics and Informatics Reports 10 (2023): 100053.

### Acknowledgements

Part of this implementation are adapted from:
- [GLFC](https://github.com/conditionWang/FCIL)
- [FOSTER](https://github.com/G-U-N/ECCV22-FOSTER)

Paper compilation:
- [Awesome-FL](https://youngfish42.github.io/Awesome-FL/)

