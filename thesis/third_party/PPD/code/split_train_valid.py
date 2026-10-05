import random

def read_data_file(file_name, train_file_name, valid_file_name):
    with open(file_name, "r") as file, open(train_file_name, "w") as train_file, open(valid_file_name, "w") as valid_file:
        for line in file:
            tokens = [int(x) for x in line.split()]
            user_id = tokens.pop(0)
            no_of_items_to_sample_for_valid = round(len(tokens) * 0.33)
            # no_of_items_to_sample_for_valid = round(len(tokens) * 0.10)
            # if splitting train (10%), then comment next two lines
            # if no_of_items_to_sample_for_valid <= 0:
            #     no_of_items_to_sample_for_valid = 1

            sample_valid_item_list = random.sample(tokens, no_of_items_to_sample_for_valid)
            sample_train_item_list = [x for x in tokens if x not in sample_valid_item_list]

            print(tokens)
            print(sample_train_item_list)
            print(sample_valid_item_list)

            sample_valid_item_list.insert(0, user_id)
            sample_train_item_list.insert(0, user_id)
            if len(sample_train_item_list) > 1:
                train_file.write(' '.join(map(str, sample_train_item_list)) + "\n")
            if len(sample_valid_item_list) > 1:
                valid_file.write(' '.join(map(str, sample_valid_item_list)) + "\n")


if __name__ == "__main__":
    random.seed(50000)
    random.seed(4200) # for KuaiRand
    read_data_file("../data/KuaiRec/train.txt",
                   "../data/KuaiRec/train_new.txt",
                   "../data/KuaiRec/valid.txt")
    # read_data_file("../data/Coat/main/test.txt",
    #                "../data/Coat/test_n.txt",
    #                "../data/Coat/valid_n.txt")