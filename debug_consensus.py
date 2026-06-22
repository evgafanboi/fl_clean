"""Compare old vs new _client_logits code paths for functional equivalence.
Run: python debug_consensus.py
Expected: max_diff should be ~0.0 if paths are truly equivalent.
"""
import os, sys, gc, tempfile, shutil
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from FL.backend import set_backend
set_backend('pt')

from models.pt_gru import PTGRUModel

input_dim, num_classes, batch_size = 78, 15, 128
n_clients, n_samples = 5, 1000
LOGIT_CLIP = 30.0

X = np.random.randn(n_samples, input_dim).astype(np.float32)
client_weights = []
for _ in range(n_clients):
    m = PTGRUModel(input_dim, num_classes, batch_size)
    client_weights.append(m.get_weights())

scratch_model = PTGRUModel(input_dim, num_classes, batch_size)

# --- Old path: freeze_task_memory style ---
# scratch_model.set_weights(weights)
# pred = scratch_model.get_logits_model().predict(X, batch_size=bs)
scratch_model.set_weights(client_weights[0])
old_pred = scratch_model.get_logits_model().predict(X, batch_size=batch_size)
old_pred = np.clip(old_pred, -LOGIT_CLIP, LOGIT_CLIP).astype(np.float32)

# --- New path: _client_logits -> _raw_logits ---
def _raw_logits(model, X_pt, bs):
    predictor = model.get_logits_model() if hasattr(model, 'get_logits_model') else model
    return predictor.predict(X_pt, batch_size=bs)

def _client_logits(scratch_model_new, cid, weights, X_pt, bs):
    model = scratch_model_new  # _scratch_pool is None
    model.set_weights(weights)
    return _raw_logits(model, X_pt, bs)

# Fresh scratch model to avoid cached state
scratch_model2 = PTGRUModel(input_dim, num_classes, batch_size)
new_pred = _client_logits(scratch_model2, 0, client_weights[0], X, batch_size)
new_pred = np.clip(new_pred, -LOGIT_CLIP, LOGIT_CLIP).astype(np.float32)

max_diff = np.abs(old_pred - new_pred).max()
print(f"max_diff = {max_diff:.10f}")
print(f"identical = {np.allclose(old_pred, new_pred, atol=1e-6)}")

# Also test: old freeze_task_memory used predict() directly (no _raw_logits)
# scratch_model.get_logits_model().predict() vs _raw_logits
scratch_model3 = PTGRUModel(input_dim, num_classes, batch_size)
scratch_model3.set_weights(client_weights[0])
raw_pred = _raw_logits(scratch_model3, X, batch_size)
raw_pred = np.clip(raw_pred, -LOGIT_CLIP, LOGIT_CLIP).astype(np.float32)
diff2 = np.abs(old_pred - raw_pred).max()
print(f"get_logits_model().predict vs _raw_logits max_diff = {diff2:.10f}")

# Test 2: set_weights + predict vs set_weights + _raw_logits
# This tests whether the model state is preserved correctly between calls
all_preds_old, all_preds_new = [], []
for cid in range(n_clients):
    scratch_model.set_weights(client_weights[cid])
    all_preds_old.append(np.clip(
        scratch_model.get_logits_model().predict(X, batch_size=batch_size),
        -LOGIT_CLIP, LOGIT_CLIP).astype(np.float32))

scratch_model4 = PTGRUModel(input_dim, num_classes, batch_size)
for cid in range(n_clients):
    all_preds_new.append(_client_logits(scratch_model4, cid, client_weights[cid], X, batch_size))

for cid in range(n_clients):
    diff = np.abs(all_preds_old[cid] - all_preds_new[cid]).max()
    print(f"client {cid}: max_diff = {diff:.10f} {'OK' if diff < 1e-5 else 'MISMATCH!'}")

print("Done.")
