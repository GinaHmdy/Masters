from collections import Counter


def read_user_item_file(path):
    """
    Reads a file where each line: user_id item1 item2 ...
    Returns: dict[user_id] = list of item_ids
    """
    data = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            user, items = parts[0], parts[1:]
            data[user] = items
    return data


def get_bottom_items(train_data, bottom_percent):
    """
    Compute bottom 80% items by popularity (frequency in train interactions).
    Returns: set of bottom 80% item ids
    """
    item_counts = Counter(item for items in train_data.values() for item in items)

    # Sort items by popularity (descending)
    sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
    n_items = len(sorted_items)

    # bottom 80% by item count
    cutoff = int(bottom_percent * n_items)
    bottom_items = set([item for item, _ in sorted_items[cutoff:]])

    return bottom_items


def filter_test_by_items(test_data, bottom_items):
    """
    Keep only user–item interactions where item ∈ valid_items
    """
    filtered = {}
    for user, items in test_data.items():
        tail_items = [i for i in items if i in bottom_items]
        if tail_items:  # keep only users with some tail interactions
            filtered[user] = tail_items
    return filtered


def write_user_item_file(path, data):
    """
    Writes user-item dict to file in format:
    user_id item1 item2 ...
    """
    with open(path, "w") as f:
        for user, items in data.items():
            line = " ".join([user] + items)
            f.write(line + "\n")


if __name__ == '__main__':
    datapath = "../data/KuaiRec/"
    train_file = datapath + "train.txt"
    test_file = datapath + "test.txt"
    bottom_percent = 0.1
    output_file = datapath + "test_bottom" + str(int(bottom_percent*100)) + ".txt"

    train_data = read_user_item_file(train_file)
    test_data = read_user_item_file(test_file)

    bottom_items = get_bottom_items(train_data, bottom_percent)
    filtered_test = filter_test_by_items(test_data, bottom_items)

    # Write filtered test set to file
    write_user_item_file(output_file, filtered_test)

    print(f"Bottom-80% test set written to {output_file}")