import torch
import numpy as np

from utils import npmse
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

# load models

#model = model_class.MLP(16, 1, args.hidden_units)
#model = model_class.RNN(hidden_units=args.hidden_units)
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
        #print(x,y)
        #raise ValueError
        optimizer.zero_grad()
        #print(x.dtype)

        loss = mse(model(x), y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.detach().cpu().item()
        print('Epoch: {}/{}, Iter: {}/{}, loss: {:.3f}'.format(epoch+1, args.epochs, i+1, len(train_loader), loss_sum/(i+1)), end='\r')
    print('')

print('Testing...')
il = []
pred = []
yl = []
xl = []
with torch.no_grad():
    loss_sum = 0.0
    for i, (x,y) in enumerate(test_loader):
        optimizer.zero_grad()
        #print(x.dtype)
        q = model(x)
        loss = mse(q, y)
        #il.append(i)
        pred.extend(q.detach().cpu().numpy())
        yl.extend(y.detach().cpu().numpy())
        xl.extend(x.detach().cpu().numpy())
        loss_sum += loss.detach().cpu().item()
        print('Iter: {}/{}, Nats: {:.3f}'.format(i+1, len(test_loader), loss_sum/(i+1)), end='\r')
    print('')
final_test_nats = loss_sum / len(test_loader)
import matplotlib.pyplot as plt
pred = np.asarray(pred)
yl = np.asarray(yl)
xl = np.asarray(xl)
minus = pred - yl
il = np.linspace(0,2500, num=2500)
z1 = (xl[:, 0] == 1)
z0 = (xl[:, 0] == 0)
#il = np.asarray(il)
print(z0)
print(yl.shape)
plt.plot(yl[z1], pred[z1], "o", label = "pred z1", markersize = 2)
plt.plot(yl[z0], pred[z0], "o", label = "pred z0", markersize = 2)
#plt.plot(il, yl,"+" ,label = "true")
plt.legend()
#plt.title("Regression Plot")
plt.xlabel("True Y value")
plt.ylabel("Predicted Y value")
plt.show()

print(np.mean(minus))

#print(xl.shape)


plt.plot(il[z1],minus[z1], "o", color = "blue", markersize = 2)
plt.title("Z1 Residual Plot")
plt.ylabel("Residual")
#plt.legend()
plt.show()

plt.plot(il[z0],minus[z0], "o", color = "orange", markersize = 2)
plt.title("Z0 Residual Plot")
plt.ylabel("Residual")
#plt.legend()
plt.show()
#print(yl[z1].shape, pred[z1].shape)
print(npmse(yl[z1], pred[z1]), npmse(yl[z0], pred[z0]))

torch.save({'epoch': epoch + 1,
            'state_dict': model.state_dict()},
           os.path.join("./results/models", 'checkpoint_{}_scale2_epoch{}.pt'.format(args.model, epoch)))

#print(z1)
#print(final_test_nats)