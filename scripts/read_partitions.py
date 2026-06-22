import os
import argparse
import numpy as np


def partition_dir(client_count: int, partition_type: str) -> str:
    return os.path.join("data", "partitions", f"{client_count}_client", partition_type)


def client_paths(client_id: int, pdir: str) -> dict:
    stem = f"client_{client_id}"
    return {
        "private_X": os.path.join(pdir, f"{stem}_X_train.npy"),
        "private_y": os.path.join(pdir, f"{stem}_y_train.npy"),
        "public_X": os.path.join(pdir, f"{stem}_X_public.npy"),
        "public_y": os.path.join(pdir, f"{stem}_y_public.npy"),
        "test_X": os.path.join(pdir, f"{stem}_X_test.npy"),
        "test_y": os.path.join(pdir, f"{stem}_y_test.npy"),
    }


def load_slice(path: str, mmap: bool = True):
    if not os.path.exists(path):
        return None
    return np.load(path, mmap_mode="r") if mmap else np.array(np.load(path))


def read_partitions(client_count: int, partition_type: str, mmap: bool = True):
    pdir = partition_dir(client_count, partition_type)
    if not os.path.isdir(pdir):
        raise FileNotFoundError(pdir)

    clients = {}
    for cid in range(client_count):
        paths = client_paths(cid, pdir)
        priv_X = load_slice(paths["private_X"], mmap)
        priv_y = load_slice(paths["private_y"], mmap)
        pub_X = load_slice(paths["public_X"], mmap)
        pub_y = load_slice(paths["public_y"], mmap)
        clients[cid] = {
            "private": {"X_path": paths["private_X"], "y_path": paths["private_y"], "X": priv_X, "y": priv_y},
            "public": {"X_path": paths["public_X"], "y_path": paths["public_y"], "X": pub_X, "y": pub_y},
            "test": {"X_path": paths["test_X"], "y_path": paths["test_y"]},
        }
    return clients


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Read client partitions and separate public/private slices")
    p.add_argument("--client-count", type=int, required=True)
    p.add_argument("--partition", type=str, required=True)
    p.add_argument("--no-mmap", dest="mmap", action="store_false")
    p.add_argument("--show-sizes", action="store_true")
    args = p.parse_args()

    clients = read_partitions(args.client_count, args.partition, mmap=args.mmap)

    if args.show_sizes:
        for cid, data in clients.items():
            priv_X = data["private"]["X"]
            pub_X = data["public"]["X"]
            priv_n = int(priv_X.shape[0]) if priv_X is not None else 0
            pub_n = int(pub_X.shape[0]) if pub_X is not None else 0
            print(f"client {cid}: private={priv_n}, public={pub_n}")
    else:
        print(f"Loaded {len(clients)} clients from {partition_dir(args.client_count, args.partition)}")
