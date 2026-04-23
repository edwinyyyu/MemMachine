import json
import sys

import matplotlib.pyplot as plt


def main():
    if len(sys.argv) < 2:
        print("Usage: total_segments_stats.py <path>")
        sys.exit(1)
    path = sys.argv[1]
    with open(path) as f:
        data = json.load(f)
    values = [item["total_segments"] for item in data]
    avg = sum(values) / len(values)
    print(f"Count: {len(values)}")
    print(f"Min total_segments: {min(values)}")
    print(f"Max total_segments: {max(values)}")
    print(f"Average total_segments: {avg:.2f}")

    plt.figure(figsize=(10, 6))
    plt.hist(
        values,
        bins=range(min(values), max(values) + 2),
        edgecolor="black",
        align="left",
    )
    plt.axvline(avg, color="red", linestyle="--", label=f"Mean: {avg:.2f}")
    plt.xlabel("Total Segments")
    plt.ylabel("Frequency")
    plt.title("Distribution of Total Segments")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
