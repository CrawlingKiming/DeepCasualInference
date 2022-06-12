import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, TensorDataset

def fetch_dataset(args):
    test = pd.read_csv("./data/test_cycle4.csv", encoding = "UTF-8")
    train = pd.read_csv("./data/train_cycle4.csv", encoding = "UTF-8")
    train_tensor = torch.tensor(train.values)
    test_tensor = torch.tensor(test.values)
    means = train_tensor.mean(dim=0, keepdim=True)
    stds = train_tensor.std(dim=0, keepdim=True)
    means[0,1] = 0
    stds[0,1]= 1
    #print(means)
    train_tensor = (train_tensor - means) / stds
    #print(train_tensor.shape)
    #print(means.shape)
    #train_tensor = train_tensor.double()
    #print(train_tensor.dtype)
    #means = test_tensor.mean(dim=0, keepdim=True)
    #stds = test_tensor.std(dim=0, keepdim=True)
    test_tensor = (test_tensor - means) / stds

    train_tensor_y = train_tensor[:, [0]]
    train_tensor_x = train_tensor[:, 1:]

    test_tensor_y = test_tensor[:, [0]]
    test_tensor_x = test_tensor[:, 1:]
    #print(torch.sum(train_tensor_x, dim=1))
    #raise ValueError

    if args.log_transform :
        train_tensor_y = torch.log(train_tensor_y + 1)
        test_tensor_y = torch.log(test_tensor_y + 1)
    dataset_train = TensorDataset(train_tensor_x, train_tensor_y)
    dataset_eval = TensorDataset(test_tensor_x, test_tensor_y)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset_eval, batch_size=args.batch_size, shuffle=True)

    return (train_loader, test_loader)

def fetch_base_dataset(args):
    test = pd.read_csv("./data/test_cycle4.csv", encoding = "UTF-8")
    train = pd.read_csv("./data/train_cycle4.csv", encoding = "UTF-8")
    train_tensor = np.asarray(train.values)
    test_tensor = np.asarray(test.values)
    means = np.mean(train_tensor, axis=0, keepdims=True)
    stds = np.std(train_tensor, axis=0, keepdims=True)
    means[0,1] = 0
    stds[0,1]= 1
    #print(means)
    train_tensor = (train_tensor - means) / stds
    #print(train_tensor.shape)
    #print(means.shape)
    #train_tensor = train_tensor.double()
    #print(train_tensor.dtype)
    #means = test_tensor.mean(dim=0, keepdim=True)
    #stds = test_tensor.std(dim=0, keepdim=True)
    test_tensor = (test_tensor - means) / stds

    #train_tensor = train_tensor.double()
    #print(train_tensor.dtype)
    train_tensor_y = train_tensor[:, [0]]
    train_tensor_x = train_tensor[:, 1:]

    test_tensor_y = test_tensor[:, [0]]
    test_tensor_x = test_tensor[:, 1:]
    if args.log_transform :
        train_tensor_y = np.log(train_tensor_y + 15)
        test_tensor_y = np.log(test_tensor_y + 15)

    return (train_tensor_x, train_tensor_y, test_tensor_x, test_tensor_y)
    #dataset_train = TensorDataset(train_tensor_x, train_tensor_y)
    #dataset_eval = TensorDataset(test_tensor_x, test_tensor_y)
    #train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    #test_loader = DataLoader(dataset_eval, batch_size=args.batch_size, shuffle=True)

#print(train_tensor_y.shape)
#print(train_tensor_y[:10, :])