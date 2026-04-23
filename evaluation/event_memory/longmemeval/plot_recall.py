import argparse
import json
import os

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("files", nargs=2, help="Two recall JSON files to compare")
args = parser.parse_args()

with open(args.files[0]) as f:
    data1 = json.load(f)
with open(args.files[1]) as f:
    data2 = json.load(f)

label1 = os.path.splitext(os.path.basename(args.files[0]))[0]
label2 = os.path.splitext(os.path.basename(args.files[1]))[0]

categories = [k for k in data1.keys() if k != "mode"]

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i, cat in enumerate(categories):
    ax = axes[i]
    r1 = data1[cat]["recall_at_k"]
    r2 = data2[cat]["recall_at_k"]
    ks = list(range(1, len(r1) + 1))

    ax.plot(ks, r1, label=label1, linewidth=1.5, marker="o", markersize=2)
    ax.plot(
        ks, r2, label=label2, linewidth=1.5, linestyle="--", marker="s", markersize=2
    )
    ax.set_title(cat)
    ax.set_xlabel("Top K")
    ax.set_ylabel("Recall")
    ax.set_xlim(1, 50)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

for j in range(len(categories), len(axes)):
    axes[j].set_visible(False)

plt.suptitle(f"Recall @ K: {label1} vs {label2}", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
