from collections import defaultdict


def find_underrepresented_items(train_file):
    # Step 1: Parse training data to get item frequencies
    item_freq = defaultdict(int)

    with open(train_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            items = parts[1:]  # skip user ID
            for item in set(items):  # count each item once per user
                item_freq[item] += 1

    # Step 2: Define underrepresented items (bottom 80%)
    sorted_items = sorted(item_freq.items(), key=lambda x: x[1])  # ascending by frequency
    cutoff_index = int(0.2 * len(sorted_items))
    underrepresented_items = set(item for item, _ in sorted_items[:cutoff_index])

    return underrepresented_items


# Step 3: Function to count how many times each item is recommended
def count_recommendation_exposure(rec_file):
    exposure = defaultdict(int)
    total_recs = 0
    with open(rec_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            items = parts[1:]  # skip user ID
            total_recs += len(items)
            for item in items:
                exposure[item] += 1
    return exposure, total_recs


# Step 5: Count underrepresented item exposures
def underrepresented_exposure(exposure_dict, under_items):
    return sum(exposure_dict[i] for i in under_items if i in exposure_dict)


if __name__ == '__main__':
    dataset = 'Yahoo'
    train_file = '../data/' + dataset + '/train.txt'
    sbd_predict_file = '../data/' + dataset + '/topk_predict_ppd.txt'
    lgn_predict_file = '../data/' + dataset + '/topk_predict_lightgcn.txt'

    underrepresented_items = find_underrepresented_items(train_file)
    # Step 4: Load recommendation data
    lgn_exp, lgn_total = count_recommendation_exposure(lgn_predict_file)
    sbd_exp, sbd_total = count_recommendation_exposure(sbd_predict_file)

    lgn_under_exp = underrepresented_exposure(lgn_exp, underrepresented_items)
    sbd_under_exp = underrepresented_exposure(sbd_exp, underrepresented_items)

    # Step 6: Compute and print results
    lgn_rate = lgn_under_exp / lgn_total
    sbd_rate = sbd_under_exp / sbd_total

    print(f"LightGCN → Underrepresented Exposure: {lgn_under_exp}/{lgn_total} = {lgn_rate:.4}")
    print(f"SBD-LightGCN → Underrepresented Exposure: {sbd_under_exp}/{sbd_total} = {sbd_rate:.4}")

    print(f"Improvements of SDB-LightGCN over LightGCN: {(sbd_rate - lgn_rate)  / (lgn_rate + 0.00001):.4}")

    if sbd_rate > lgn_rate:
        print("✅ SBD improves exposure to underrepresented items.")
    else:
        print("❌ SBD does not improve exposure to underrepresented items.")

    print(f"{lgn_rate:.4f} & {sbd_rate:.4f} & {(sbd_rate - lgn_rate)  / (lgn_rate + 0.000001):+.2f}")