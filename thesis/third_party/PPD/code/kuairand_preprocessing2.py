import pandas as pd


def map_all_item_ids(train_file, test_file):
    df = pd.read_csv(train_file)
    df = df[['user_id', 'video_id', 'is_click']]
    df = df.sort_values(by='user_id', ascending=True)

    all_clicked_video_ids_set_train = set()
    item_id_mapping = {}

    for user_id, user_df in df.groupby('user_id'):
        video_ids = user_df['video_id'].tolist()
        is_clicks = user_df['is_click'].tolist()

        # getting the clicked video ids
        for i in range(len(is_clicks)):
            if is_clicks[i] == 1:
                all_clicked_video_ids_set_train.add(video_ids[i])

    all_clicked_video_ids_list_train = list(all_clicked_video_ids_set_train)
    all_clicked_video_ids_list_train.sort()

    for i in range(len(all_clicked_video_ids_list_train)):
        item_id_mapping[all_clicked_video_ids_list_train[i]] = i


    df = pd.read_csv(test_file)
    df = df[['user_id', 'video_id', 'is_click']]
    df = df.sort_values(by='user_id', ascending=True)

    all_clicked_video_ids_set_test = set()

    for user_id, user_df in df.groupby('user_id'):
        video_ids = user_df['video_id'].tolist()
        is_clicks = user_df['is_click'].tolist()

        # getting the clicked video ids
        for i in range(len(is_clicks)):
            if is_clicks[i] == 1:
                all_clicked_video_ids_set_test.add(video_ids[i])

    all_clicked_video_ids_list_test = list(all_clicked_video_ids_set_test)
    all_clicked_video_ids_list_test.sort()
    print(item_id_mapping)

    index = len(all_clicked_video_ids_list_train)
    for i in range(len(all_clicked_video_ids_list_test)):
        if all_clicked_video_ids_list_test[i] not in item_id_mapping:
            item_id_mapping[all_clicked_video_ids_list_test[i]] = index
            index += 1

    print(item_id_mapping)

    return item_id_mapping



def read_dataset_train(read_file, write_file, item_id_mapping):
    df = pd.read_csv(read_file)
    df = df[['user_id', 'video_id', 'is_click']]
    df = df.sort_values(by='user_id', ascending=True)

    all_clicked_video_ids_set = set()
    all_interacted_user_ids_set = set()
    incremental_user_id_counter = 0
    user_id_mapping = {}

    with open(write_file, "w") as output_file:
        for user_id, user_df in df.groupby('user_id'):
            user_id_mapping[user_id] = incremental_user_id_counter
            clicked_video_ids = []
            video_ids = user_df['video_id'].tolist()
            is_clicks = user_df['is_click'].tolist()

            # getting the clicked video ids
            for i in range(len(is_clicks)):
                if is_clicks[i] == 1:
                    video_id_from_mapping = item_id_mapping[video_ids[i]]
                    clicked_video_ids.append(video_id_from_mapping)
                    all_clicked_video_ids_set.add(video_id_from_mapping)

            # writing the clicked video ids to file
            if len(clicked_video_ids) > 0:
                all_interacted_user_ids_set.add(incremental_user_id_counter)
                clicked_video_ids.insert(0, incremental_user_id_counter)
                output_file.write(' '.join(map(str, clicked_video_ids)) + "\n")
                incremental_user_id_counter += 1

        print("Number of interacted users in train: " + str(len(all_clicked_video_ids_set)))
        print("Number of interacted videos in train: " + str(len(all_interacted_user_ids_set)))

        return all_clicked_video_ids_set, user_id_mapping


def read_dataset_test(read_file, write_file, train_videos, user_id_mapping_train, item_id_mapping):
    df = pd.read_csv(read_file)
    df = df[['user_id', 'video_id', 'is_click']]
    df = df.sort_values(by='user_id', ascending=True)

    all_clicked_video_ids_set = set()
    all_interacted_user_ids_set = set()
    c = 0
    with open(write_file, "w") as output_file:
        for user_id, user_df in df.groupby('user_id'):
            if user_id in user_id_mapping_train:
                user_id_from_mapping = user_id_mapping_train[user_id]
            else:
                c += 1
                continue

            clicked_video_ids = []
            video_ids = user_df['video_id'].tolist()
            is_clicks = user_df['is_click'].tolist()

            # getting the clicked video ids
            for i in range(len(is_clicks)):
                if is_clicks[i] == 1 and item_id_mapping[video_ids[i]] in train_videos:
                    video_id_from_mapping = item_id_mapping[video_ids[i]]
                    clicked_video_ids.append(video_id_from_mapping)
                    all_clicked_video_ids_set.add(video_id_from_mapping)

            # writing the clicked video ids to file
            if len(clicked_video_ids) > 0:
                all_interacted_user_ids_set.add(user_id_from_mapping)
                clicked_video_ids.insert(0, user_id_from_mapping)
                output_file.write(' '.join(map(str, clicked_video_ids)) + "\n")

        print("Number of interacted users in test: " + str(len(all_clicked_video_ids_set)))
        print("Number of interacted videos in test: " + str(len(all_interacted_user_ids_set)))
        print(c)

        return all_interacted_user_ids_set, all_clicked_video_ids_set


if __name__ == "__main__":
    file1 = '../data/KuaiRand/log_standard_4_08_to_4_21_pure.csv'
    file2 = '../data/KuaiRand/train.txt'

    file3 = '../data/KuaiRand/log_random_4_22_to_5_08_pure.csv'
    file4 = '../data/KuaiRand/test.txt'

    item_id_mappings = map_all_item_ids(file1, file3)
    train_videos_ret, user_id_mappings_train = read_dataset_train(file1, file2, item_id_mappings)
    read_dataset_test(file3, file4, train_videos_ret, user_id_mappings_train, item_id_mappings)