import torch
import numpy as np
import pandas as pd

import os
import argparse

# Optimizer
from torch.optim import Adam, Adamax

# Custom modules
import dataloading
import model_class

parser = argparse.ArgumentParser()

# Model params
parser.add_argument('--hidden_units', type=eval, default=[100,100,100, 100])
parser.add_argument('--model', type=str, default='MLP', choices={'MLP', 'RNN', "RNN_FULL"})

parser.add_argument('--activation', type=str, default='relu', choices={'relu', 'elu', 'gelu'})
parser.add_argument('--range_flow', type=str, default='logit', choices={'logit', 'softplus'})
parser.add_argument('--log_transform', type=eval, default=False)

# Train params
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--optimizer', type=str, default='adam', choices={'adam', 'adamax'})
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--device', type=str, default='cpu')
args = parser.parse_args()

# Set your device
args.device = torch.device('cuda:{}'.format(0))

# First load datasets
train_loader, test_loader = dataloading.fetch_dataset(args)

# load pretrained models
pretrained_model = model_class.model_fetcher(args)
pretrained_model = pretrained_model.double()

epoch = 49
model_path = os.path.join("./results/models", 'checkpoint_{}_scale2_epoch{}.pt'.format("RNN", epoch))
checkpoint = torch.load(model_path)
pretrained_model.load_state_dict(checkpoint["state_dict"])

# load models

args.model = "MLP"
model = model_class.model_fetcher(args)
model = model.double()

if args.optimizer == 'adam':
    optimizer = Adam(model.parameters(), lr=args.lr)
elif args.optimizer == 'adamax':
    optimizer = Adamax(model.parameters(), lr=args.lr)

print('Training...')
mse = torch.nn.MSELoss()
for epoch in range(args.epochs):
    loss_sum = 0.0
    for i, (x,y) in enumerate(train_loader):
        optimizer.zero_grad()
        #print(x.dtype)
        z1 = (x[:, 0] > 0.5)
        x[z1, 0] = 0
        x[~z1, 0] = 1
        #print(x[:, 0] > 0.5)

        with torch.no_grad():
            imputed_y = pretrained_model(x)
        # should be Y(z=1) - Y(z=0)
        diff_y = imputed_y - y
        #print(diff_y)
        mask = -1 * torch.ones(size=diff_y.size())
        mask[x[:, 0] > 0.5] = 0
        #print(mask)
        ACE = diff_y * mask
        #print(ACE)
        loss = mse(model(x), ACE)
        loss.backward()
        optimizer.step()
        loss_sum += loss.detach().cpu().item()
        print('Epoch: {}/{}, Iter: {}/{}, loss: {:.3f}'.format(epoch+1, args.epochs, i+1, len(train_loader), loss_sum/(i+1)), end='\r')
    print('')

print('Testing...')
il = []
pred = []
yl = []
with torch.no_grad():
    loss_sum = 0.0
    for i, (x,y) in enumerate(test_loader):
        optimizer.zero_grad()
        #print(x.dtype)
        z1 = (x[:, 0] > 0.5)
        x[z1, 0] = 0
        x[~z1, 0] = 1
        #print(x[:, 0] > 0.5)

        with torch.no_grad():
            imputed_y = pretrained_model(x)
        # should be Y(z=1) - Y(z=0)
        diff_y = imputed_y - y
        #print(diff_y)
        mask = -1 * torch.ones(size=diff_y.size())
        mask[x[:, 0] > 0.5] = 0
        #print(mask)
        ACE = diff_y * mask

        q = model(x)
        loss = mse(q, ACE)
        #il.append(i)
        pred.extend(q.detach().cpu().numpy())
        yl.extend(ACE.detach().cpu().numpy())
        loss_sum += loss.detach().cpu().item()
        print('Iter: {}/{}, Nats: {:.3f}'.format(i+1, len(test_loader), loss_sum/(i+1)), end='\r')
    print('')
final_test_nats = loss_sum / len(test_loader)
import matplotlib.pyplot as plt
pred = np.asarray(pred)
yl = np.asarray(yl)
minus = pred - yl
il = np.linspace(0,2500, num=2500)
#il = np.asarray(il)
#print(yl.shape)
plt.plot(yl, pred, "o", label = "pred", color = "orange", markersize = 2)
#plt.plot(il, yl,"+" ,label = "true")
plt.legend()
#plt.title("Regression Plot")
plt.xlabel("Y(1) - Y(0) with imputation")
plt.ylabel("Predicted Y(1) - Y(0)")
plt.show()

plt.plot(il,minus, "o", color = "orange", markersize = 2)
plt.title("Residual Plot")
plt.ylabel("Residual")
plt.legend()
plt.show()
print(np.mean(minus))


#print(final_test_nats)