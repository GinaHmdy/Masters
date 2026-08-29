"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Shuxian Bi (stanbi@mail.ustc.edu.cn),Jianbai Ye (gusye@mail.ustc.edu.cn)
Design Dataset here
Every dataset's index has to start at 0
"""
import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time

class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")

    @property
    def n_users(self):
        raise NotImplementedError

    @property
    def m_items(self):
        raise NotImplementedError

    @property
    def trainDataSize(self):
        raise NotImplementedError

    @property
    def testDict(self):
        raise NotImplementedError

    @property
    def validDict(self):
        raise NotImplementedError

    @property
    def allPos(self):
        raise NotImplementedError

    def getUserItemFeedback(self, users, items):
        raise NotImplementedError

    def getUserPosItems(self, users):
        raise NotImplementedError

    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError


    def getSparseGraph(self, variant="lgn"):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

class LastFM(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    LastFM dataset
    """
    def __init__(self, path="../../data/lastfm"):
        # train or test
        cprint("loading [last fm]")
        self.mode_dict = {'train':0, "test":1}
        self.mode    = self.mode_dict['train']
        # self.n_users = 1892
        # self.m_items = 4489
        trainData = pd.read_table(join(path, 'data1.txt'), header=None)
        # print(trainData.head())
        testData  = pd.read_table(join(path, 'test1.txt'), header=None)
        # print(testData.head())
        trustNet  = pd.read_table(join(path, 'trustnetwork.txt'), header=None).to_numpy()
        # print(trustNet[:5])
        trustNet -= 1
        trainData-= 1
        testData -= 1
        self.trustNet  = trustNet
        self.trainData = trainData
        self.testData  = testData
        self.trainUser = np.array(trainData[:][0])
        self.trainUniqueUsers = np.unique(self.trainUser)
        self.trainItem = np.array(trainData[:][1])
        # self.trainDataSize = len(self.trainUser)
        self.testUser  = np.array(testData[:][0])
        self.testUniqueUsers = np.unique(self.testUser)
        self.testItem  = np.array(testData[:][1])
        self.Graph = None
        print(f"LastFm Sparsity : {(len(self.trainUser) + len(self.testUser))/self.n_users/self.m_items}")

        # (users,users)
        self.socialNet    = csr_matrix((np.ones(len(trustNet)), (trustNet[:,0], trustNet[:,1]) ), shape=(self.n_users,self.n_users))
        # (users,items), bipartite graph
        self.UserItemNet  = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem) ), shape=(self.n_users,self.m_items))

        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_users)))
        self.allNeg = []
        allItems    = set(range(self.m_items))
        for i in range(self.n_users):
            pos = set(self._allPos[i])
            neg = allItems - pos
            self.allNeg.append(np.array(list(neg)))
        self.__testDict = self.__build_test()

    @property
    def n_users(self):
        return 1892

    @property
    def m_items(self):
        return 4489

    @property
    def trainDataSize(self):
        return len(self.trainUser)

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def getSparseGraph(self, variant="lgn"):
        if self.Graph is None:
            user_dim = torch.LongTensor(self.trainUser)
            item_dim = torch.LongTensor(self.trainItem)

            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            second_sub = torch.stack([item_dim+self.n_users, user_dim])
            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).int()
            self.Graph = torch.sparse.IntTensor(index, data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            dense = self.Graph.to_dense()
            D = torch.sum(dense, dim=1).float()
            D[D==0.] = 1.
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
            dense = dense/D_sqrt
            dense = dense/D_sqrt.t()
            index = dense.nonzero()
            data  = dense[dense >= 1e-9]
            assert len(index) == len(data)
            self.Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            self.Graph = self.Graph.coalesce().to(world.device)
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1, ))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def getUserNegItems(self, users):
        negItems = []
        for user in users:
            negItems.append(self.allNeg[user])
        return negItems



    def __getitem__(self, index):
        user = self.trainUniqueUsers[index]
        # return user_id and the positive items of the user
        return user

    def switch2test(self):
        """
        change dataset mode to offer test data to dataloader
        """
        self.mode = self.mode_dict['test']

    def __len__(self):
        return len(self.trainUniqueUsers)

class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self,config = world.config,path="../../data/Gowalla"):
        # train or test or valid
        cprint(f'loading [{path}]')
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1, "valid": 2}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        valid_file = path + '/valid.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        validUniqueUsers, validItem, validUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0
        self.validDataSize = 0

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)

        with open(valid_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip().split(' ')
                    if len(l)<=1:
                        continue
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    validUniqueUsers.append(uid)
                    validUser.extend([uid] * len(items))
                    validItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.validDataSize += len(items)

        self.m_item += 1
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        self.validUniqueUsers = np.array(validUniqueUsers)
        self.validUser = np.array(validUser)
        self.validItem = np.array(validItem)

        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{self.validDataSize} interactions for validation")
        print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize + self.validDataSize) / self.n_users / self.m_items}")

        if config.get('denoise_edges', False):
            print("[Denoising] Running IsolationForest to denoise user-item interactions...")
            u_degrees = np.bincount(self.trainUser, minlength=self.n_user)
            i_degrees = np.bincount(self.trainItem, minlength=self.m_item)
            features = np.column_stack((u_degrees[self.trainUser], i_degrees[self.trainItem]))
            
            from sklearn.ensemble import IsolationForest
            iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
            preds = iso.fit_predict(features)
            keep_mask = (preds == 1)
            pruned_count = len(keep_mask) - keep_mask.sum()
            print(f"[Denoising] Pruned {pruned_count} anomalous interactions out of {len(self.trainUser)} ({pruned_count/len(self.trainUser)*100:.2f}%)")
            
            self.trainUser = self.trainUser[keep_mask]
            self.trainItem = self.trainItem[keep_mask]
            self.traindataSize = len(self.trainUser)

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        self.__validDict = self.__build_valid()
        print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def validDict(self):
        return self.__validDict

    def _candidate_universe(self, split):
        """
        Items a given split is actually able to score, or None for no restriction.

        KuaiRec's splits are drawn from the dense fully-observed "small matrix", which
        covers only ~30% of the catalogue. Ranking over all m_items lets the model spend
        its top-K on items that appear nowhere in the evaluated split and therefore can
        never be a hit, which drives every metric below random. Restricting the candidate
        set to the evaluated universe is the standard KuaiRec protocol. Other datasets
        cover >94% of the catalogue and are left unrestricted so their published numbers
        stay comparable.

        The universe is derived PER SPLIT: using the test universe while selecting on
        validation would leak which items are testable into model selection.
        """
        cache = getattr(self, "_candidate_cache", None)
        if cache is None:
            cache = self._candidate_cache = {}
        if split in cache:
            return cache[split]
        if not world.config.get("restrict_candidates", False):
            cache[split] = None
            return None
        source = {"test": self.__testDict, "valid": self.__validDict}[split]
        universe = set()
        for items in source.values():
            universe.update(items)
        idx = np.array(sorted(universe), dtype=np.int64)
        print(f"[eval] candidate set restricted to {len(idx)} / {self.m_items} items "
              f"({split}-split universe)")
        cache[split] = idx
        return idx

    @property
    def candidateItems(self):
        """Candidate universe for the TEST split. See _candidate_universe."""
        return self._candidate_universe("test")

    @property
    def validCandidateItems(self):
        """Candidate universe for the VALIDATION split. See _candidate_universe."""
        return self._candidate_universe("valid")

    @property
    def allPos(self):
        raise NotImplementedError

    def getUserItemFeedback(self, users, items):
        raise NotImplementedError

    def getUserPosItems(self, users):
        raise NotImplementedError

    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError


    def getSparseGraph(self, variant="lgn"):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

class LastFM(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    LastFM dataset
    """
    def __init__(self, path="../../data/lastfm"):
        # train or test
        cprint("loading [last fm]")
        self.mode_dict = {'train':0, "test":1}
        self.mode    = self.mode_dict['train']
        # self.n_users = 1892
        # self.m_items = 4489
        trainData = pd.read_table(join(path, 'data1.txt'), header=None)
        # print(trainData.head())
        testData  = pd.read_table(join(path, 'test1.txt'), header=None)
        # print(testData.head())
        trustNet  = pd.read_table(join(path, 'trustnetwork.txt'), header=None).to_numpy()
        # print(trustNet[:5])
        trustNet -= 1
        trainData-= 1
        testData -= 1
        self.trustNet  = trustNet
        self.trainData = trainData
        self.testData  = testData
        self.trainUser = np.array(trainData[:][0])
        self.trainUniqueUsers = np.unique(self.trainUser)
        self.trainItem = np.array(trainData[:][1])
        # self.trainDataSize = len(self.trainUser)
        self.testUser  = np.array(testData[:][0])
        self.testUniqueUsers = np.unique(self.testUser)
        self.testItem  = np.array(testData[:][1])
        self.Graph = None
        print(f"LastFm Sparsity : {(len(self.trainUser) + len(self.testUser))/self.n_users/self.m_items}")

        # (users,users)
        self.socialNet    = csr_matrix((np.ones(len(trustNet)), (trustNet[:,0], trustNet[:,1]) ), shape=(self.n_users,self.n_users))
        # (users,items), bipartite graph
        self.UserItemNet  = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem) ), shape=(self.n_users,self.m_items))

        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_users)))
        self.allNeg = []
        allItems    = set(range(self.m_items))
        for i in range(self.n_users):
            pos = set(self._allPos[i])
            neg = allItems - pos
            self.allNeg.append(np.array(list(neg)))
        self.__testDict = self.__build_test()

    @property
    def n_users(self):
        return 1892

    @property
    def m_items(self):
        return 4489

    @property
    def trainDataSize(self):
        return len(self.trainUser)

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def getSparseGraph(self, variant="lgn"):
        if self.Graph is None:
            user_dim = torch.LongTensor(self.trainUser)
            item_dim = torch.LongTensor(self.trainItem)

            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            second_sub = torch.stack([item_dim+self.n_users, user_dim])
            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).int()
            self.Graph = torch.sparse.IntTensor(index, data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            dense = self.Graph.to_dense()
            D = torch.sum(dense, dim=1).float()
            D[D==0.] = 1.
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
            dense = dense/D_sqrt
            dense = dense/D_sqrt.t()
            index = dense.nonzero()
            data  = dense[dense >= 1e-9]
            assert len(index) == len(data)
            self.Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            self.Graph = self.Graph.coalesce().to(world.device)
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1, ))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def getUserNegItems(self, users):
        negItems = []
        for user in users:
            negItems.append(self.allNeg[user])
        return negItems



    def __getitem__(self, index):
        user = self.trainUniqueUsers[index]
        # return user_id and the positive items of the user
        return user

    def switch2test(self):
        """
        change dataset mode to offer test data to dataloader
        """
        self.mode = self.mode_dict['test']

    def __len__(self):
        return len(self.trainUniqueUsers)

class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self,config = world.config,path="../../data/Gowalla"):
        # train or test or valid
        cprint(f'loading [{path}]')
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1, "valid": 2}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        valid_file = path + '/valid.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        validUniqueUsers, validItem, validUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0
        self.validDataSize = 0

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)

        with open(valid_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip().split(' ')
                    if len(l)<=1:
                        continue
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    validUniqueUsers.append(uid)
                    validUser.extend([uid] * len(items))
                    validItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.validDataSize += len(items)

        self.m_item += 1
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        self.validUniqueUsers = np.array(validUniqueUsers)
        self.validUser = np.array(validUser)
        self.validItem = np.array(validItem)

        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{self.validDataSize} interactions for validation")
        print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize + self.validDataSize) / self.n_users / self.m_items}")

        if config.get('denoise_edges', False):
            print("[Denoising] Running IsolationForest to denoise user-item interactions...")
            u_degrees = np.bincount(self.trainUser, minlength=self.n_user)
            i_degrees = np.bincount(self.trainItem, minlength=self.m_item)
            features = np.column_stack((u_degrees[self.trainUser], i_degrees[self.trainItem]))
            
            from sklearn.ensemble import IsolationForest
            iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
            preds = iso.fit_predict(features)
            keep_mask = (preds == 1)
            pruned_count = len(keep_mask) - keep_mask.sum()
            print(f"[Denoising] Pruned {pruned_count} anomalous interactions out of {len(self.trainUser)} ({pruned_count/len(self.trainUser)*100:.2f}%)")
            
            self.trainUser = self.trainUser[keep_mask]
            self.trainItem = self.trainItem[keep_mask]
            self.traindataSize = len(self.trainUser)

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        self.__validDict = self.__build_valid()
        print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def validDict(self):
        return self.__validDict

    def _candidate_universe(self, split):
        """
        Items a given split is actually able to score, or None for no restriction.

        KuaiRec's splits come from the dense fully-observed "small matrix", covering only
        ~30% of the catalogue. Ranking over all m_items lets the model spend its top-K on
        items that appear nowhere in the evaluated split and therefore can never be a hit,
        which drives every metric below random. Restricting candidates to the evaluated
        universe is the standard KuaiRec protocol. Other datasets cover >94% of the
        catalogue and stay unrestricted so their published numbers remain comparable.

        The universe is derived PER SPLIT: reusing the test universe while selecting on
        validation would leak which items are testable into model selection.
        """
        cache = getattr(self, "_candidate_cache", None)
        if cache is None:
            cache = self._candidate_cache = {}
        if split in cache:
            return cache[split]
        if not world.config.get("restrict_candidates", False):
            cache[split] = None
            return None
        source = self.__testDict if split == "test" else self.__validDict
        universe = set()
        for items in source.values():
            universe.update(items)
        idx = np.array(sorted(universe), dtype=np.int64)
        print(f"[eval] candidate set restricted to {len(idx)} / {self.m_items} items "
              f"({split}-split universe)")
        cache[split] = idx
        return idx

    @property
    def candidateItems(self):
        """Candidate universe for the TEST split. See _candidate_universe."""
        return self._candidate_universe("test")

    @property
    def validCandidateItems(self):
        """Candidate universe for the VALIDATION split. See _candidate_universe."""
        return self._candidate_universe("valid")

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self, variant="lgn"):
        print("loading adjacency matrix")
        if self.Graph is None:
            if world.config.get('denoise_edges', False):
                pre_adj_mat = None
            else:
                try:
                    pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                    print("successfully loaded...")
                    norm_adj = pre_adj_mat
                except :
                    pre_adj_mat = None
            if pre_adj_mat is None:
                print("generating adjacency matrix")
                s = time()
                R = self.UserItemNet
                zero_u = sp.csr_matrix((self.n_users, self.n_users), dtype=np.float32)
                zero_i = sp.csr_matrix((self.m_items, self.m_items), dtype=np.float32)
                
                if variant == "navip":
                    # Compute item interaction-based weighting
                    item_interaction_counts = self.UserItemNet.sum(axis=0).A1  # Convert to 1D array
                    item_interaction_counts[item_interaction_counts == 0] = 1  # Prevent division by zero
                    max_popular_item_interaction = item_interaction_counts.max()

                    normalized_ipw_for_navip = 1 / (
                            np.sqrt(item_interaction_counts / max_popular_item_interaction) + 1e-6)
                    ipw_min = normalized_ipw_for_navip.min()
                    ipw_max = normalized_ipw_for_navip.max()
                    normalized_ipw_for_navip = (normalized_ipw_for_navip - normalized_ipw_for_navip.min()) / (ipw_max - ipw_min + 1e-6)

                    # Create diagonal weighting matrix for items
                    item_weight_diag = sp.diags(normalized_ipw_for_navip)

                    # Apply weighting using CSR dot multiplication
                    R_weighted = R.dot(item_weight_diag)
                    adj_mat = sp.bmat([[zero_u, R_weighted], [R_weighted.T, zero_i]], format='csr', dtype=np.float32)
                else:
                    adj_mat = sp.bmat([[zero_u, R], [R.T, zero_i]], format='csr', dtype=np.float32)

                rowsum = np.array(adj_mat.sum(axis=1)).flatten()
                d_inv = np.zeros_like(rowsum)
                non_zero_mask = rowsum > 0
                d_inv[non_zero_mask] = np.power(rowsum[non_zero_mask], -0.5)
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                if not world.config.get('denoise_edges', False):
                    sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("don't split the matrix")
        return self.Graph


    # Added by Aminul
    def getSparseGraphWithWeight(self, bias_scores):
        print("loading adjacency matrix")

        print("generating adjacency matrix")
        s = time()
        adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        # R = self.UserItemNet.tolil()

        # variant 1
        R = self.UserItemNet.tolil()
        for u in range(self.n_users):
            index = 0
            for i in R.rows[u]:  # only iterate over interacted items
                R[u, i] = 1 - (bias_scores[u][index] + 1e-9)  # apply the custom weight
                index += 1

        # R = self.UserItemNet.tolil()
        # for u in range(self.n_users):
        #     row_items = R.rows[u]  # items user u interacted with
        #     new_values = []
        #
        #     # Step 1: Compute unnormalized bias-weighted values
        #     for index, i in enumerate(row_items):
        #         weight = 1 / (bias_scores[u][index] + 1e-9)
        #         new_values.append(weight)
        #
        #     # Step 2: Normalize weights (sum-to-1 normalization)
        #     total = sum(new_values)
        #     if total == 0:
        #         normalized_values = [0 for _ in new_values]
        #     else:
        #         normalized_values = [v / total for v in new_values]
        #
        #     # Step 3: Assign normalized weights back to R[u, i]
        #     for i, val in zip(row_items, normalized_values):
        #         R[u, i] = val

        # # Step 0: Prepare to collect all weights
        # R = self.UserItemNet.tolil()
        # all_weights = []  # for global min-max normalization
        # user_item_pairs = []  # to keep track of (u, i, weight) to reassign later
        #
        # # Step 1: Compute inverse bias weights and store them
        # for u in range(self.n_users):
        #     row_items = R.rows[u]
        #     for index, i in enumerate(row_items):
        #         weight = 1 / (bias_scores[u][index] + 1e-9)
        #         all_weights.append(weight)
        #         user_item_pairs.append((u, i, weight))
        #
        # # Step 2: Min-max normalize all weights
        # all_weights = np.array(all_weights)
        # min_val = all_weights.min()
        # max_val = all_weights.max()
        # range_val = max_val - min_val if max_val != min_val else 1e-9
        # normalized_weights = (all_weights - min_val) / range_val
        #
        R = R.tocsr()
        zero_u = sp.csr_matrix((self.n_users, self.n_users), dtype=np.float32)
        zero_i = sp.csr_matrix((self.m_items, self.m_items), dtype=np.float32)
        adj_mat = sp.bmat([[zero_u, R], [R.T, zero_i]], format='csr', dtype=np.float32)

        rowsum = np.array(adj_mat.sum(axis=1)).flatten()
        d_inv = np.zeros_like(rowsum)
        non_zero_mask = rowsum > 0
        d_inv[non_zero_mask] = np.power(rowsum[non_zero_mask], -0.5)
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_mat).dot(d_mat)
        end = time()
        print(f"costing {end - s}s, saved norm_mat...")
        # sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

        if self.split == True:
            self.Graph = self._split_A_hat(norm_adj)
            print("done split matrix")
        else:
            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(world.device)
            print("don't split the matrix")
        # print(self.Graph[0])
        return self.Graph


    def getOnlyGraphParams(self):
        return self.Graph


    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data


    def __build_valid(self):
        """
        return:
            dict: {user: [items]}
        """
        valid_data = {}
        for i, item in enumerate(self.validItem):
            user = self.validUser[i]
            if valid_data.get(user):
                valid_data[user].append(item)
            else:
                valid_data[user] = [item]
        return valid_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    # def getUserNegItems(self, users):
    #     negItems = []
    #     for user in users:
    #         negItems.append(self.allNeg[user])
    #     return negItems


from scipy.stats import rankdata


class PopularityDensityStore:
    """
    Computes and caches popularity and density features for items and users.
    Ensures rank-normalized features in [0, 1] across datasets.
    """
    def __init__(self, train_matrix, n_users, m_items, cache_dir=".cache", dataset_name="dataset"):
        self.n_users = n_users
        self.m_items = m_items
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir

        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = os.path.join(self.cache_dir, f"{dataset_name}_popularity_density.pt")

        if os.path.exists(cache_file):
            try:
                data = torch.load(cache_file)
                if data["item_popularity_rank"].shape[0] == self.m_items and data["user_popularity"].shape[0] == self.n_users:
                    self.item_degree = data["item_degree"]
                    self.item_popularity_raw = data["item_popularity_raw"]
                    self.item_popularity_rank = data["item_popularity_rank"]
                    self.user_popularity = data["user_popularity"]
                    self.item_density_raw = data["item_density_raw"]
                    self.item_density_rank = data["item_density_rank"]
                    self.user_density_raw = data["user_density_raw"]
                    self.user_density_rank = data["user_density_rank"]
                    self.item_popularity_decile = data["item_popularity_decile"]
                    self.item_bucket_indices = data["item_bucket_indices"]
                    self.user_history_decile_diversity = data["user_history_decile_diversity"]
                    print(f"[PopularityDensityStore] Loaded cached features from {cache_file}")
                    return
                else:
                    print(f"[PopularityDensityStore] Cached dimension mismatch ({data['item_popularity_rank'].shape[0]} items vs expected {self.m_items}), recomputing...")
            except Exception as e:
                print(f"[PopularityDensityStore] Cache read error: {e}, recomputing...")

        self._compute_features(train_matrix)

        # Save cache
        try:
            torch.save({
                "item_degree": self.item_degree,
                "item_popularity_raw": self.item_popularity_raw,
                "item_popularity_rank": self.item_popularity_rank,
                "user_popularity": self.user_popularity,
                "item_density_raw": self.item_density_raw,
                "item_density_rank": self.item_density_rank,
                "user_density_raw": self.user_density_raw,
                "user_density_rank": self.user_density_rank,
                "item_popularity_decile": self.item_popularity_decile,
                "item_bucket_indices": self.item_bucket_indices,
                "user_history_decile_diversity": self.user_history_decile_diversity
            }, cache_file)
            print(f"[PopularityDensityStore] Saved features cache to {cache_file}")
        except Exception as e:
            print(f"[PopularityDensityStore] Could not save cache: {e}")

    def _compute_features(self, R):
        if not sp.issparse(R):
            R = sp.csr_matrix(R)
        else:
            R = R.tocsr()

        # 1. Item degree and raw popularity
        item_deg = np.array(R.sum(axis=0)).flatten().astype(np.float32)
        self.item_degree = torch.from_numpy(item_deg)
        self.item_popularity_raw = torch.log1p(self.item_degree)

        # Rank-normalized item popularity in [0, 1]
        if self.m_items > 1:
            ranks = rankdata(item_deg, method="average") - 1.0
            item_pop_rank = (ranks / (self.m_items - 1.0)).astype(np.float32)
        else:
            item_pop_rank = np.zeros(self.m_items, dtype=np.float32)
        self.item_popularity_rank = torch.from_numpy(item_pop_rank)

        # 2. User degree and user popularity (average of interacted item popularity)
        user_deg = np.array(R.sum(axis=1)).flatten().astype(np.float32)
        user_pop = np.zeros(self.n_users, dtype=np.float32)

        R_csc = R.tocsc()
        for u in range(self.n_users):
            items_u = R.indices[R.indptr[u]:R.indptr[u+1]]
            if len(items_u) > 0:
                user_pop[u] = np.mean(item_pop_rank[items_u])
            else:
                user_pop[u] = 0.5
        self.user_popularity = torch.from_numpy(user_pop)

        # 3. Item 2nd-order density: c_i = sum_j sum_u R_ui R_uj = sum_u R_ui * deg_u
        user_deg_vec = user_deg
        c_i = np.array(R_csc.T.dot(user_deg_vec)).flatten().astype(np.float32)
        self.item_density_raw = torch.log1p(torch.from_numpy(c_i))
        if self.m_items > 1:
            item_dens_ranks = (rankdata(c_i, method="average") - 1.0) / (self.m_items - 1.0)
        else:
            item_dens_ranks = np.zeros(self.m_items, dtype=np.float32)
        self.item_density_rank = torch.from_numpy(item_dens_ranks.astype(np.float32))

        # 4. User density: delta_u = log(1 + sum_{i in N(u)} deg_i)
        user_dens_raw = np.zeros(self.n_users, dtype=np.float32)
        for u in range(self.n_users):
            items_u = R.indices[R.indptr[u]:R.indptr[u+1]]
            if len(items_u) > 0:
                user_dens_raw[u] = np.sum(item_deg[items_u])
        self.user_density_raw = torch.log1p(torch.from_numpy(user_dens_raw))
        if self.n_users > 1:
            user_dens_ranks = (rankdata(user_dens_raw, method="average") - 1.0) / (self.n_users - 1.0)
        else:
            user_dens_ranks = np.zeros(self.n_users, dtype=np.float32)
        self.user_density_rank = torch.from_numpy(user_dens_ranks.astype(np.float32))

        # 5. Item popularity deciles (0 to 9) and 5-quantile bucket indices for basis SVD
        deciles = np.clip((item_pop_rank * 10.0).astype(np.int64), 0, 9)
        self.item_popularity_decile = torch.from_numpy(deciles)

        # 5 buckets for subspace basis candidate construction
        quantiles_5 = np.clip((item_pop_rank * 5.0).astype(np.int64), 0, 4)
        self.item_bucket_indices = torch.from_numpy(quantiles_5)

        # 6. User diversity feature: unique deciles in user history / 10.0
        user_diversity = np.zeros(self.n_users, dtype=np.float32)
        for u in range(self.n_users):
            items_u = R.indices[R.indptr[u]:R.indptr[u+1]]
            if len(items_u) > 0:
                u_deciles = set(deciles[items_u])
                user_diversity[u] = len(u_deciles) / 10.0
            else:
                user_diversity[u] = 0.1
        self.user_history_decile_diversity = torch.from_numpy(user_diversity)


class StratifiedPopularityNegativeSampler:
    """
    Stratified mixed negative sampler with logQ correction.
    Draws negatives from uniform, popularity-matched, and top-decile hard negatives.
    Gracefully falls back without throwing exceptions or dropping training examples.
    """
    def __init__(
        self,
        item_popularity_rank,
        item_popularity_decile,
        all_pos_dict,
        num_items,
        uniform_ratio=0.4,
        matched_ratio=0.4,
        hard_ratio=0.2,
    ):
        self.num_items = num_items
        self.all_pos_dict = all_pos_dict
        
        # Normalize mixture ratios to sum to 1.0
        total_r = uniform_ratio + matched_ratio + hard_ratio
        if total_r <= 0:
            uniform_ratio, matched_ratio, hard_ratio = 1.0, 0.0, 0.0
            total_r = 1.0
        self.r_u = uniform_ratio / total_r
        self.r_m = matched_ratio / total_r
        self.r_h = hard_ratio / total_r

        if isinstance(item_popularity_rank, torch.Tensor):
            self.pop_rank = item_popularity_rank.cpu().numpy()
        else:
            self.pop_rank = np.array(item_popularity_rank)

        if isinstance(item_popularity_decile, torch.Tensor):
            self.deciles = item_popularity_decile.cpu().numpy()
        else:
            self.deciles = np.array(item_popularity_decile)

        # Precompute item pools for each decile
        self.decile_items = [np.where(self.deciles == d)[0] for d in range(10)]
        for d in range(10):
            if len(self.decile_items[d]) == 0:
                self.decile_items[d] = np.arange(self.num_items)

        # Hard negative bucket (top decile: decile 9 or top 10% items)
        self.hard_items = self.decile_items[9]
        if len(self.hard_items) == 0:
            top_k = max(1, int(0.1 * self.num_items))
            self.hard_items = np.argsort(self.pop_rank)[-top_k:]

        self.decile_sizes = np.array([len(self.decile_items[d]) for d in range(10)], dtype=np.float32)
        self.hard_size = float(len(self.hard_items))

    def sample(self, user_ids, pos_item_ids, num_negatives=1):
        """
        Samples negatives for a batch of (user_id, pos_item_id).
        Returns:
            negative_item_ids: [B, num_negatives] (numpy or tensor)
            logq: [B, num_negatives] sampling log-probabilities for correction
        """
        batch_size = len(user_ids)
        neg_items = np.zeros((batch_size, num_negatives), dtype=np.int64)
        neg_logq = np.zeros((batch_size, num_negatives), dtype=np.float32)
        eps = 1e-8

        for b in range(batch_size):
            u = int(user_ids[b])
            pos_i = int(pos_item_ids[b])
            if isinstance(self.all_pos_dict, dict):
                pos_set = self.all_pos_dict.get(u, set())
            elif u < len(self.all_pos_dict):
                pos_set = self.all_pos_dict[u]
            else:
                pos_set = set()
            if isinstance(pos_set, (list, np.ndarray)):
                pos_set = set(pos_set)
            
            pos_decile = self.deciles[pos_i] if pos_i < self.num_items else 0
            matched_pool = self.decile_items[pos_decile]

            for k in range(num_negatives):
                rand_val = np.random.rand()
                selected_item = -1

                # 1. Stratified selection
                if rand_val < self.r_u or len(matched_pool) == 0:
                    # Uniform sampling
                    for _ in range(50):
                        cand = np.random.randint(0, self.num_items)
                        if cand not in pos_set:
                            selected_item = cand
                            break
                    if selected_item == -1:
                        selected_item = np.random.randint(0, self.num_items)

                elif rand_val < self.r_u + self.r_m:
                    # Popularity-matched sampling
                    for _ in range(50):
                        cand = np.random.choice(matched_pool)
                        if cand not in pos_set:
                            selected_item = cand
                            break
                    if selected_item == -1:
                        # Fallback to uniform
                        for _ in range(50):
                            cand = np.random.randint(0, self.num_items)
                            if cand not in pos_set:
                                selected_item = cand
                                break
                        if selected_item == -1:
                            selected_item = np.random.randint(0, self.num_items)

                else:
                    # Hard top-decile sampling
                    for _ in range(50):
                        cand = np.random.choice(self.hard_items)
                        if cand not in pos_set:
                            selected_item = cand
                            break
                    if selected_item == -1:
                        # Fallback to matched or uniform
                        cand = np.random.choice(matched_pool)
                        if cand not in pos_set:
                            selected_item = cand
                        else:
                            selected_item = np.random.randint(0, self.num_items)

                neg_items[b, k] = selected_item

                # 2. Compute exact logQ probability: q_j = r_u * q_unif + r_m * q_matched + r_h * q_hard
                cand_decile = self.deciles[selected_item] if selected_item < self.num_items else 0
                q_unif = 1.0 / max(1, self.num_items)
                q_matched = (1.0 / max(1.0, self.decile_sizes[pos_decile])) if (cand_decile == pos_decile) else 0.0
                q_hard = (1.0 / max(1.0, self.hard_size)) if (selected_item in self.hard_items) else 0.0

                q_total = self.r_u * q_unif + self.r_m * q_matched + self.r_h * q_hard
                neg_logq[b, k] = np.log(max(q_total, eps))

        return neg_items, neg_logq

