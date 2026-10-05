import matplotlib.pyplot as plt
import numpy as np

from collections import Counter, defaultdict
import torch


def draw_item_interaction_distributions(input_file_path, output_file_path):
    # Step 1: Read data and build item-to-users mapping
    item_to_users = defaultdict(set)

    with open(input_file_path, "r") as f:
        for line in f:
            parts = list(map(int, line.strip().split()))
            user_id = parts[0]
            item_ids = parts[1:]
            for item in item_ids:
                item_to_users[item].add(user_id)

    # Step 2: Compute number of unique users per item
    users_per_item = [len(users) for users in item_to_users.values()]
    users_per_item_sorted = sorted(users_per_item, reverse=True)

    # Step 3: Beautiful plot setup
    # plt.figure(figsize=(14, 10), frameon=True)
    fig, ax = plt.subplots(figsize=(14, 10), frameon=True)
    ax.set_facecolor('#f0f8ff')  # light blue tone
    plt.plot(users_per_item_sorted, color="#1f77b4", linewidth=5)

    plt.xlabel("Items (sorted by number of interactions)", fontsize=36)
    plt.ylabel("Number of users", fontsize=36)

    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36)
    plt.tight_layout()
    plt.box(True)
    plt.savefig(output_file_path)


def draw_head_mid_tail(input_file_path, output_file_path):
    all_items = []
    with open(input_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            all_items.extend(parts[1:])

    item_counts = Counter(all_items)
    sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)

    n = len(sorted_items)

    top_items = sorted_items[:n//3]
    mid_items = sorted_items[n//3:2*n//3]
    tail_items = sorted_items[2*n//3:]

    top_sum = sum(count for _, count in top_items)
    mid_sum = sum(count for _, count in mid_items)
    tail_sum = sum(count for _, count in tail_items)

    total = top_sum + mid_sum + tail_sum
    percentages = [top_sum / total * 100, mid_sum / total * 100, tail_sum / total * 100]

    # Set group positions and labels
    x = [0, 1, 2]
    labels = ['Head', 'Mid', 'Tail']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    # colors = ['orange', 'orchid', 'green', 'cyan', 'cyan', 'purple']

    # Plot
    plt.figure(figsize=(14, 10), frameon=True)
    bars = plt.bar(x, percentages, color=colors)
    plt.xticks(x, fontsize=36)
    plt.xticks(ticks=range(len(x)), labels=labels)

    plt.ylabel('Percentage of interactions (%)', fontsize=36)
    plt.xlabel('Item popularity group', fontsize=36)
    # plt.title('Interaction Percentage by Item Popularity Group', fontsize=15)
    plt.ylim(0, 100)
    plt.yticks(np.arange(0, 101, 20), fontsize=36)

    # Annotate bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{percentages[i]:.1f}%', ha='center', fontsize=36)

    # plt.box(False)
    plt.savefig(output_file_path)


def draw_head_tail(datasets, labels, output_file_path):
    """
    Draw stacked bars of head (top 20%) and tail (bottom 80%) interactions
    for multiple datasets.

    datasets: list of file paths to input datasets
    labels: list of dataset names (for x-axis)
    output_file_path: path to save figure
    """
    results = []  # list of (head_pct, tail_pct) for each dataset

    for input_file_path in datasets:
        all_items = []
        with open(input_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                all_items.extend(parts[1:])

        # Count item frequencies
        item_counts = Counter(all_items)
        sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)

        n = len(sorted_items)
        head_items = sorted_items[:max(1, int(n * 0.2))]  # top 20%
        tail_items = sorted_items[int(n * 0.2):]  # bottom 80%

        head_sum = sum(count for _, count in head_items)
        tail_sum = sum(count for _, count in tail_items)
        total = head_sum + tail_sum

        head_pct = head_sum / total * 100
        tail_pct = tail_sum / total * 100
        results.append((head_pct, tail_pct))

    # Plot stacked bars with shadows
    x = np.arange(len(datasets))
    width = 0.6

    head_vals = [r[0] for r in results]
    tail_vals = [r[1] for r in results]

    plt.figure(figsize=(14, 10), frameon=True)
    bars1 = plt.bar(
        x, head_vals, width,
        label='Head items (Top 20%)',
        color='peru',
        zorder=3
    )
    bars2 = plt.bar(
        x, tail_vals, width,
        bottom=head_vals,
        label='Tail items (Bottom 80%)',
        color='#2ca02c',
        zorder=3
    )

    plt.xticks(x, labels, fontsize=32)
    plt.ylabel('Percentage of interactions (%)', fontsize=32)
    plt.xlabel('Datasets', fontsize=32)
    plt.ylim(0, 115)  # extra space above 100%
    plt.yticks(np.arange(0, 101, 20), fontsize=32)

    # Grid for depth effect
    # plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

    # Annotate bars
    for i, (h, t) in enumerate(zip(head_vals, tail_vals)):
        plt.text(i, h / 2, f'{h:.1f}%', ha='center', va='center', fontsize=24, color='white')
        plt.text(i, h + t / 2, f'{t:.1f}%', ha='center', va='center', fontsize=24, color='white')

    # Legend INSIDE the figure, using actual bar handles
    plt.legend(
        handles=[bars1, bars2],
        fontsize=26,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        frameon=True
    )

    plt.tight_layout()
    plt.savefig(output_file_path, dpi=300, bbox_inches='tight')
    plt.close()


def read_user_interactions(file_path: str) -> list[list[int]]:
    user_interactions = []
    with open(file_path, 'r') as f:
        for line in f:
            items = list(map(int, line.strip().split()))
            user_interactions.append(items)
    return user_interactions


def compute_popularity_kl(user_interactions: list[list[int]], num_items: int) -> float:
    # Step 1: Count total item interactions
    item_counts = Counter()
    for items in user_interactions:
        item_counts.update(items)

    # Step 2: Convert to tensor and normalize to get empirical distribution P
    P = torch.zeros(num_items)
    for i in range(num_items):
        P[i] = item_counts[i]

    epsilon = 1e-10  # to avoid log(0)
    P = P + epsilon
    P = P / P.sum()

    # Step 3: Define uniform distribution Q
    Q = torch.full_like(P, 1.0 / num_items)

    # Step 4: Compute KL divergence: KL(P || Q)
    kl_div = torch.sum(P * (P.log() - Q.log()))
    return kl_div.item()


def measure_kl_divergence(file_path):
    # Assume 'interactions.txt' has each user's item list on a line
    user_interactions = read_user_interactions(file_path)

    # Determine max item index to set num_items
    all_items = [item for user in user_interactions for item in user]
    num_items = max(all_items) + 1  # assuming item indices start at 0

    # Compute KL divergence
    kl = compute_popularity_kl(user_interactions, num_items)
    print("KL Divergence (Popularity Bias):", kl)


def compute_entropy(file_path) -> float:
    user_interactions = read_user_interactions(file_path)
    # Flatten all item interactions
    all_items = [item for user in user_interactions for item in user]

    # Determine number of unique items
    item_set = set(all_items)
    item_to_idx = {item: idx for idx, item in enumerate(sorted(item_set))}
    num_items = len(item_set)

    # Count item frequencies
    item_counts = torch.zeros(num_items)
    for item in all_items:
        idx = item_to_idx[item]
        item_counts[idx] += 1

    # Convert to probability distribution
    epsilon = 1e-10
    P = item_counts + epsilon
    P = P / P.sum()

    # Compute normalized entropy
    entropy = -torch.sum(P * P.log()) / torch.log(torch.tensor(num_items, dtype=torch.float))
    print("Entropy of items:", entropy)
    return entropy.item()


if __name__ == "__main__":

    # datasets = ["../data/KuaiRec/train.txt", "../data/Coat/train.txt", "../data/Yahoo/train.txt"]
    # labels = ["KuaiRec", "Coat", "Yahoo! R3"]
    # draw_head_tail(datasets, labels, "head_tail_comparison.pdf")

    input_file_path = "../data/KuaiRec/train.txt"
    output_file_path = "../data/KuaiRec/train_head_mid_tail_KuaiRec.pdf"
    draw_head_mid_tail(input_file_path, output_file_path)

    input_file_path = "../data/Coat/train.txt"
    output_file_path = "../data/Coat/train_head_mid_tail_Coat.pdf"
    draw_head_mid_tail(input_file_path, output_file_path)

    input_file_path = "../data/Yahoo/train.txt"
    output_file_path = "../data/Yahoo/train_head_mid_tail_Yahoo.pdf"
    draw_head_mid_tail(input_file_path, output_file_path)

    input_file_path = "../data/KuaiRec/train.txt"
    output_file_path = "../data/KuaiRec/train_item_interaction_dist_KuaiRec.pdf"
    draw_item_interaction_distributions(input_file_path, output_file_path)

    input_file_path = "../data/Coat/train.txt"
    output_file_path = "../data/Coat/train_item_interaction_dist_Yahoo.pdf"
    draw_item_interaction_distributions(input_file_path, output_file_path)

    input_file_path = "../data/Yahoo/train.txt"
    output_file_path = "../data/Yahoo/train_item_interaction_dist_Yahoo.pdf"
    draw_item_interaction_distributions(input_file_path, output_file_path)

    measure_kl_divergence("../data/KuaiRec/train.txt")
    measure_kl_divergence("../data/KuaiRec/test.txt")
    measure_kl_divergence("../data/KuaiRand/train.txt")
    measure_kl_divergence("../data/KuaiRand/test.txt")
    measure_kl_divergence("../data/Coat/train.txt")
    measure_kl_divergence("../data/Coat/test.txt")

    compute_entropy("../data/KuaiRec/train.txt")
    compute_entropy("../data/KuaiRec/test.txt")
    compute_entropy("../data/KuaiRand/train.txt")
    compute_entropy("../data/KuaiRand/test.txt")
    compute_entropy("../data/Coat/train.txt")
    compute_entropy("../data/Coat/test.txt")