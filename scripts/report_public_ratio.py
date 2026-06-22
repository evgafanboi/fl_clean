import os
import sys
import argparse
import numpy as np

# Make `scripts` imports work when running this file directly.
script_dir = os.path.dirname(__file__) or "."
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
try:
    from read_partitions import read_partitions, partition_dir
except Exception:
    repo_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from scripts.read_partitions import read_partitions, partition_dir

OUT_DIR = os.path.join("temp_weights")
OUT_FILE = os.path.join(OUT_DIR, "a.md")


def format_ratio(pub, priv):
    if priv is None and pub is None:
        return "N/A"
    p = int(pub) if pub is not None else 0
    q = int(priv) if priv is not None else 0
    m = p + q
    if m == 0:
        return "N/A"
    return f"{p}/{m} ({p/m:.4f})"


def write_report(clients, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lines = []
    lines.append("# Public to Merged Ratio Report\n")
    lines.append("| client_id | private_count | public_count | merged_count | public_ratio |")
    lines.append("|---:|---:|---:|---:|---:|")

    total_priv = 0
    total_pub = 0
    n_clients = 0

    for cid, data in sorted(clients.items()):
        priv = data["private"]["X"]
        pub = data["public"]["X"]
        priv_n = int(priv.shape[0]) if priv is not None else 0
        pub_n = int(pub.shape[0]) if pub is not None else 0
        merged = priv_n + pub_n
        ratio = f"{pub_n}/{merged} ({pub_n/merged:.6f})" if merged > 0 else "N/A"
        lines.append(f"| {cid} | {priv_n} | {pub_n} | {merged} | {ratio} |")
        total_priv += priv_n
        total_pub += pub_n
        n_clients += 1

    overall_merged = total_priv + total_pub
    overall_ratio = f"{total_pub}/{overall_merged} ({total_pub/overall_merged:.6f})" if overall_merged > 0 else "N/A"
    lines.append("\n")
    lines.append(f"**Clients:** {n_clients}  ")
    lines.append(f"**Total private samples:** {total_priv}  ")
    lines.append(f"**Total public samples:** {total_pub}  ")
    lines.append(f"**Overall public/merged ratio:** {overall_ratio}  ")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote report to {out_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--client-count", type=int, required=True)
    p.add_argument("--partition", type=str, required=True)
    p.add_argument("--no-mmap", dest="mmap", action="store_false")
    args = p.parse_args()

    clients = read_partitions(args.client_count, args.partition, mmap=args.mmap)
    write_report(clients, OUT_FILE)
    print("Done.")
