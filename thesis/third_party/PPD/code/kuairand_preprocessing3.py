

def read_and_convert(train_file, test_file, valid_file, train_file_w, test_file_w, valid_file_w):
    user_ids_set = set()
    item_ids_set = set()
    with open(train_file, "r") as file_t:
        for line in file_t:
            items = [int(x) for x in line.split()]
            user_id = items.pop(0)
            user_ids_set.add(user_id)
            for item in items:
                item_ids_set.add(item)

    user_ids_list = list(user_ids_set)
    user_ids_list.sort()
    item_ids_list = list(item_ids_set)
    item_ids_list.sort()

    user_id_mappings = {}
    item_id_mappings = {}

    for i in range(len(user_ids_list)):
        user_id_mappings[user_ids_list[i]] = i
    for i in range(len(item_ids_list)):
        item_id_mappings[item_ids_list[i]] = i

    with open(train_file, "r") as file, open(train_file_w, "w") as train_write:
        for line in file:
            write_list_train = []
            items = [int(x) for x in line.split()]
            user_id = items.pop(0)
            user_id_from_mapping = user_id_mappings[user_id]
            write_list_train.append(user_id_from_mapping)
            for item in items:
                item_id_from_mapping = item_id_mappings[item]
                write_list_train.append(item_id_from_mapping)
            train_write.write(' '.join(map(str, write_list_train)) + "\n")

    with open(test_file, "r") as file, open(test_file_w, "w") as test_write:
        for line in file:
            write_list_test = []
            items = [int(x) for x in line.split()]
            user_id = items.pop(0)
            user_id_from_mapping = user_id_mappings[user_id]
            write_list_test.append(user_id_from_mapping)
            for item in items:
                item_id_from_mapping = item_id_mappings[item]
                write_list_test.append(item_id_from_mapping)
            test_write.write(' '.join(map(str, write_list_test)) + "\n")

    with open(valid_file, "r") as file, open(valid_file_w, "w") as valid_write:
        for line in file:
            write_list_valid = []
            items = [int(x) for x in line.split()]
            user_id = items.pop(0)
            user_id_from_mapping = user_id_mappings[user_id]
            write_list_valid.append(user_id_from_mapping)
            for item in items:
                item_id_from_mapping = item_id_mappings[item]
                write_list_valid.append(item_id_from_mapping)
            valid_write.write(' '.join(map(str, write_list_valid)) + "\n")


if __name__ == "__main__":
    file_1 = "../data/KuaiRand/train.txt"
    file_2 = "../data/KuaiRand/test.txt"
    file_3 = "../data/KuaiRand/valid.txt"

    file_4 = "../data/KuaiRand/train1.txt"
    file_5 = "../data/KuaiRand/test1.txt"
    file_6 = "../data/KuaiRand/valid1.txt"

    read_and_convert(file_1, file_2, file_3, file_4, file_5, file_6)