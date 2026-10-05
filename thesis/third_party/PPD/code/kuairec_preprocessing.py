import pandas as pd


def read_dataset_train(read_file, write_file):
    df = pd.read_csv(read_file)
    df = df[['user_id', 'video_id', 'watch_ratio']]
    df = df.sort_values(by='user_id', ascending=True)

    all_clicked_video_ids_set = set()
    all_interacted_user_ids_set = set()
    no_of_interactions = 0

    with open(write_file, "w") as output_file:
        for user_id, user_df in df.groupby('user_id'):
            clicked_video_ids = []
            video_ids = user_df['video_id'].tolist()
            watch_ratios = user_df['watch_ratio'].tolist()

            # getting the clicked video ids
            for i in range(len(watch_ratios)):
                if watch_ratios[i] > 2.0:
                    clicked_video_ids.append(video_ids[i])
                    all_clicked_video_ids_set.add(video_ids[i])
                    no_of_interactions += 1

            # writing the clicked video ids to file
            if len(clicked_video_ids) > 0:
                all_interacted_user_ids_set.add(user_id)
                clicked_video_ids.insert(0, user_id)
                output_file.write(' '.join(map(str, clicked_video_ids)) + "\n")

        print("Number of interacted users in train: " + str(len(all_clicked_video_ids_set)))
        print("Number of interacted videos in train: " + str(len(all_interacted_user_ids_set)))
        print("Number of interactions in train: " + str(no_of_interactions))

        return all_interacted_user_ids_set, all_clicked_video_ids_set


def read_dataset_test(read_file, write_file, train_users, train_videos):
    df = pd.read_csv(read_file)
    df = df[['user_id', 'video_id', 'watch_ratio']]
    df = df.sort_values(by='user_id', ascending=True)

    all_clicked_video_ids_set = set()
    all_interacted_user_ids_set = set()
    # c = 0
    no_of_interactions = 0

    with open(write_file, "w") as output_file:
        for user_id, user_df in df.groupby('user_id'):
            if user_id not in train_users:
                # c += 1
                continue
            clicked_video_ids = []
            video_ids = user_df['video_id'].tolist()
            watch_ratios = user_df['watch_ratio'].tolist()

            # getting the clicked video ids
            for i in range(len(watch_ratios)):
                if watch_ratios[i] > 2.0 and video_ids[i] in train_videos:
                    clicked_video_ids.append(video_ids[i])
                    all_clicked_video_ids_set.add(video_ids[i])
                    no_of_interactions += 1

            # writing the clicked video ids to file
            if len(clicked_video_ids) > 0:
                all_interacted_user_ids_set.add(user_id)
                clicked_video_ids.insert(0, user_id)
                output_file.write(' '.join(map(str, clicked_video_ids)) + "\n")

        print("Number of interacted users in test: " + str(len(all_clicked_video_ids_set)))
        print("Number of interacted videos in test: " + str(len(all_interacted_user_ids_set)))
        print("Number of interactions in test: " + str(no_of_interactions))
        # print(c)

        return all_interacted_user_ids_set, all_clicked_video_ids_set


if __name__ == "__main__":
    file1 = '../data/KuaiRec/big_matrix.csv'
    file2 = '../data/KuaiRec/train.txt'

    file3 = '../data/KuaiRec/small_matrix.csv'
    file4 = '../data/KuaiRec/test.txt'

    train_users_ret, train_videos_ret = read_dataset_train(file1, file2)
    read_dataset_test(file3, file4, train_users_ret, train_videos_ret)