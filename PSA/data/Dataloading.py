import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import normalize

#data = normalize(data, axis=0, norm='max')
def fetch_dataset(args):

    if args.dataset == "PSA" or "pros_cancer":
        train = np.load("PSA_train_{}.npy".format(args.rate))
        test = np.load("PSA_valid.npy")
    else :
        raise NotImplementedError

    #train
    train_x = train[:, 1:]
    train_y = train[:, [0]]
    #print(torch.sum(train_y))
    #pp = train_x.max(axis=0)
    #train_x = train_x / train_x.max(axis=0)
    train_x = torch.tensor(train_x)
    train_y = torch.tensor(train_y)

    print(torch.sum(train_y))
    dataset_train = TensorDataset(train_x, train_y)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)

    #test
    test_x = test[:, 1:]
    test_y = test[:, [0]]
    #test_x = test_x / pp
    test_x = torch.tensor(test_x)
    test_y = torch.tensor(test_y)

    dataset_test = TensorDataset(test_x, test_y)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    return (train_loader, test_loader)

def fetch_base_dataset(args):
    #test = pd.read_csv("./data/test_cycle4.csv", encoding = "UTF-8")
    #train = pd.read_csv("./data/train_cycle4.csv", encoding = "UTF-8")
    rate = args.rate

    #np.load("PSA_train_1000.npy")
    #np.load("P")
    test_np = np.load("PSA_valid.npy")
    train_np = np.load("PSA_train_{}.npy".format(rate))


    return (train_np, test_np)