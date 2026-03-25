#!/usr/bin/env python3
"""Pre-create merged partition files so the runtime merge is a no-op.

The partitioner (partition_data.py) carves each client's data into private
``client_{i}_X_train.npy`` and public ``client_{i}_X_public.npy`` slices.
Strategies that don't use the public split (FedAvg, FedDyn, FedProx, …)
merge them back at runtime via ``restore_full_partition()``, which writes
``_merged.npy`` files every run — wasteful on constrained setups like Colab.

This script pre-creates those ``*_X_merged.npy`` / ``*_y_merged.npy`` files
on disk so the runtime function finds them already present.  No original
files are modified or deleted.

Usage
-----
# Process a specific partition dir
python restore_partition.py data/partitions/100_client/label_skew_0.1

# Process every partition dir under data/partitions/
python restore_partition.py data/partitions/ --all

# Preview without writing anything
python restore_partition.py data/partitions/ --all --dry-run

# Remove previously generated _merged files
python restore_partition.py data/partitions/100_client/label_skew_0.1 --clean
"""

import argparse
import glob
import os
import re
import sys

import numpy as np


def detect_clients(partition_dir: str):
    """Return sorted list of client IDs that have both train and public files."""
    pattern = os.path.join(partition_dir, "client_*_X_train.npy")
    train_files = glob.glob(pattern)
    client_ids = []
    for f in train_files:
        m = re.search(r"client_(\d+)_X_train\.npy$", f)
        if m:
            cid = int(m.group(1))
            pub = os.path.join(partition_dir, f"client_{cid}_X_public.npy")
            if os.path.exists(pub):
                client_ids.append(cid)
    return sorted(client_ids)


def create_merged(partition_dir: str, dry_run: bool = False, force: bool = False) -> int:
    """Create _merged.npy files from train + public.  Returns number of clients processed."""
    client_ids = detect_clients(partition_dir)
    if not client_ids:
        print(f"  skip  {partition_dir} (no public splits found)")
        return 0

    created = 0
    for cid in client_ids:
        x_merged = os.path.join(partition_dir, f"client_{cid}_X_merged.npy")
        y_merged = os.path.join(partition_dir, f"client_{cid}_y_merged.npy")

        if not force and os.path.exists(x_merged) and os.path.exists(y_merged):
            print(f"  exists client {cid}  (use --force to recreate)")
            continue

        if dry_run:
            print(f"  [dry-run] would create merged files for client {cid}")
            created += 1
            continue

        X_train = np.load(os.path.join(partition_dir, f"client_{cid}_X_train.npy"))
        y_train = np.load(os.path.join(partition_dir, f"client_{cid}_y_train.npy"))
        X_pub = np.load(os.path.join(partition_dir, f"client_{cid}_X_public.npy"))
        y_pub = np.load(os.path.join(partition_dir, f"client_{cid}_y_public.npy"))

        np.save(x_merged, np.concatenate([X_train, X_pub], axis=0))
        np.save(y_merged, np.concatenate([y_train, y_pub], axis=0))

        created += 1
        print(f"  ✓ client {cid}  ({X_train.shape[0]}+{X_pub.shape[0]} = {X_train.shape[0]+X_pub.shape[0]} samples)")

    print(f"  {created} clients in {partition_dir}")
    return created


def clean_merged(partition_dir: str) -> int:
    """Remove _merged.npy files."""
    removed = 0
    for f in sorted(glob.glob(os.path.join(partition_dir, "client_*_merged.npy"))):
        os.remove(f)
        removed += 1
    if removed:
        print(f"  removed {removed} merged files from {partition_dir}")
    else:
        print(f"  skip  {partition_dir} (no merged files)")
    return removed


def find_partition_dirs(root: str):
    """Yield directories that contain client partition .npy files."""
    for dirpath, _dirnames, filenames in os.walk(root):
        if any(f.startswith("client_") and f.endswith("_X_train.npy") for f in filenames):
            yield dirpath


def main():
    parser = argparse.ArgumentParser(
        description="Pre-create _merged.npy files (train + public) so runtime merge is a no-op.",
    )
    parser.add_argument("path", help="Partition directory, or root when used with --all")
    parser.add_argument("--all", action="store_true", help="Recursively process every partition dir under PATH")
    parser.add_argument("--clean", action="store_true", help="Remove _merged.npy files instead of creating them")
    parser.add_argument("--force", action="store_true", help="Recreate merged files even if they already exist")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without writing files")
    args = parser.parse_args()

    if not os.path.isdir(args.path):
        print(f"Error: {args.path} is not a directory", file=sys.stderr)
        sys.exit(1)

    dirs = list(find_partition_dirs(args.path)) if args.all else [args.path]
    if not dirs:
        print(f"No partition directories found under {args.path}")
        sys.exit(0)

    total = 0
    for d in sorted(dirs):
        print(f"\n{'[clean]' if args.clean else '[merge]'} {d}")
        if args.clean:
            total += clean_merged(d)
        else:
            total += create_merged(d, dry_run=args.dry_run, force=args.force)

    label = "merged files removed" if args.clean else "clients merged"
    if args.dry_run:
        label = "clients would be merged"
    print(f"\nDone. {total} {label}.")


if __name__ == "__main__":
    main()
