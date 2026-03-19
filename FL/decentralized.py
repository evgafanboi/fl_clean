import hashlib
import logging
import random
from typing import Dict, List

import numpy as np

from .colors import COLORS


def _hash_weights(weights: List[np.ndarray]) -> int:
    h = hashlib.sha256()
    for w in weights:
        h.update(w.tobytes())
    return int(h.hexdigest(), 16)


def braintorrent_select_server(weights: List[np.ndarray], n_clients: int) -> int:
    seed = _hash_weights(weights) % (2**32)
    rng = random.Random(seed)
    return rng.randint(0, n_clients - 1)


def log_braintorrent_selection(
    round_num: int,
    selected_server: int,
    n_clients: int,
    logger: logging.Logger,
):
    msg = f"[BrainTorrent] Round {round_num}: Client {selected_server} selected as server (out of {n_clients})"
    logger.info(msg)
    print(f"{COLORS.OKCYAN}{msg}{COLORS.ENDC}")


def compute_model_similarity_scores(
    client_weights_list: List[List[np.ndarray]],
    global_weights: List[np.ndarray],
    participating_clients: List[int],
) -> Dict[int, float]:
    global_flat = np.concatenate([w.flatten() for w in global_weights])
    norm_g = np.linalg.norm(global_flat)
    scores = {}
    for client_id, client_weights in zip(participating_clients, client_weights_list):
        client_flat = np.concatenate([w.flatten() for w in client_weights])
        norm_c = np.linalg.norm(client_flat)
        if norm_c > 0 and norm_g > 0:
            scores[client_id] = float(np.dot(client_flat, global_flat) / (norm_c * norm_g))
        else:
            scores[client_id] = 0.0
    return scores


def select_model_similarity_server(scores: Dict[int, float]) -> int:
    return max(scores, key=lambda k: scores[k])


def log_model_similarity_selection(
    round_num: int,
    scores: Dict[int, float],
    selected_server: int,
    logger: logging.Logger,
):
    header = f"[ModelSimilarity] Round {round_num} server selection (similarity from round {round_num - 1}):"
    logger.info(header)
    print(f"{COLORS.OKCYAN}{header}{COLORS.ENDC}")
    for client_id in sorted(scores):
        line = f"  Client {client_id}: {scores[client_id]:.4f}"
        logger.info(line)
        print(f"{COLORS.OKCYAN}{line}{COLORS.ENDC}")
    result = f"  Most similar: Client {selected_server} (score: {scores[selected_server]:.4f}) → server for round {round_num}"
    logger.info(result)
    print(f"{COLORS.OKCYAN}{result}{COLORS.ENDC}")
