import sys; sys.path.insert(0, '.')
import argparse
import numpy as np
import FL.backend as _b; _b.set_backend(False)
from FL.strategy.robust_filter import RobustFilterV3

parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, default=1000, help="public samples")
parser.add_argument("--K", type=int, default=10,   help="total clients")
parser.add_argument("--C", type=int, default=4,    help="number of classes")
parser.add_argument("--byz_ratio",  type=float, default=0.1,  help="fraction Byzantine")
parser.add_argument("--min_ratio",  type=float, default=0.15, help="fraction legitimate minority")
parser.add_argument("--conf",       type=float, default=0.85, help="dominant-class confidence")
parser.add_argument("--noise",      type=float, default=0.02, help="prediction noise std")
parser.add_argument("--robust_threshold", type=float, default=0.7)
args = parser.parse_args()

N, K, C = args.N, args.K, args.C
n_byz = max(1, round(K * args.byz_ratio))
n_min = max(1, round(K * args.min_ratio))
n_maj = K - n_byz - n_min
assert n_maj > 0, "byz_ratio + min_ratio must be < 1"

rng = np.random.RandomState(42)

def make_preds(N, n_clients, dominant_class, C, conf, noise, rng):
    S = np.full((N, n_clients, C), (1 - conf) / (C - 1), dtype=np.float32)
    S[:, :, dominant_class] = conf
    S += rng.normal(0, noise, S.shape).astype(np.float32)
    S = np.clip(S, 1e-6, 1.0)
    S /= S.sum(axis=-1, keepdims=True)
    return S

maj_class = 0
# minority and byzantine target: the class majority predicts least (argmin of majority mean)
maj_mean = np.full(C, (1 - args.conf) / (C - 1))
maj_mean[maj_class] = args.conf
tgt_class = int(np.argmin(maj_mean))  # = any non-dominant class, e.g. 1

S = np.concatenate([
    make_preds(N, n_maj, maj_class, C, args.conf, args.noise, rng),
    make_preds(N, n_byz, tgt_class, C, args.conf, args.noise, rng),
    make_preds(N, n_min, tgt_class, C, args.conf, args.noise, rng),
], axis=1)

roles = ['majority'] * n_maj + ['byzantine'] * n_byz + ['minority'] * n_min

rf = RobustFilterV3(robust_threshold=args.robust_threshold)
discard_counts, _ = rf.count_discards(S)
fracs = discard_counts / N

print(f"\nSetup: K={K}  C={C}  N={N}  majority={n_maj}  byzantine={n_byz}  minority={n_min}")
print(f"Majority target: class {maj_class}   Minority/Byzantine target: class {tgt_class}")
print(f"RobustFilterV3  robust_threshold={args.robust_threshold}\n")
print(f"{'Client':>6}  {'Role':<10}  {'Discard%':>8}  {'Excluded':>8}")
for i, (role, f) in enumerate(zip(roles, fracs)):
    print(f"  {i:3d}    {role:<10}  {f:>8.4f}  {'YES' if f > rf.robust_threshold else 'no':>8}")

survivors = fracs <= rf.robust_threshold
survivor_roles = [r for r, s in zip(roles, survivors) if s]
final_mean = S[:, survivors, :].mean(axis=(0, 1))
print(f"\nSurvivors: {np.where(survivors)[0].tolist()}  ({survivor_roles})")
print(f"Final mean: " + "  ".join(f"class{c}={final_mean[c]:.4f}" for c in range(C)))

byz_survived = sum(1 for r, s in zip(roles, survivors) if r == 'byzantine' and s)
print(f"\nByzantine clients in final aggregation: {byz_survived}/{n_byz}")
