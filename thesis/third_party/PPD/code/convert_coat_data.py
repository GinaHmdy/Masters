

if __name__ == "__main__":
    with open('../data/Coat/main/train.ascii', 'r', encoding='ascii') as f, open("../data/Coat/main/train.txt", "w") as outfile:
        no_of_interactions = 0
        user_id = 0
        for line in f:
            int_list = list(map(int, line.split()))
            indexes = [str(i) for i, x in enumerate(int_list) if x >= 3]
            no_of_interactions += len(indexes)
            indexes.insert(0, str(user_id))
            if len(indexes) > 1:
                # print((indexes))
                outfile.write(" ".join(indexes) + "\n")
            user_id += 1
        print("Total number of train interactions: {}".format(no_of_interactions))
        print("Total number of users in train: {}".format(user_id))


    with open('../data/Coat/main/test.ascii', 'r', encoding='ascii') as f, open("../data/Coat/main/test.txt", "w") as outfile:
        no_of_interactions = 0
        user_id = 0
        for line in f:
            int_list = list(map(int, line.split()))
            indexes = [str(i) for i, x in enumerate(int_list) if x >= 3]
            no_of_interactions += len(indexes)
            indexes.insert(0, str(user_id))
            # print((indexes))
            if len(indexes) > 1:
                # print(indexes)
                outfile.write(" ".join(indexes) + "\n")
            user_id += 1
        print("Total number of test interactions: {}".format(no_of_interactions))
        print("Total number of users in test: {}".format(user_id))