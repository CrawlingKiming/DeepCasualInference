import torch
import numpy as np

import os
import argparse
#from utils import npmse

# Custom modules

#import data.Dataloading
#from .data.Dataloading import fetch_base_dataset
#import model_class
from Dataloading import fetch_base_dataset

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

parser = argparse.ArgumentParser()

# Model params
parser.add_argument('--hidden_units', type=eval, default=[100,100,100, 100])
parser.add_argument('--activation', type=str, default='relu', choices={'relu', 'elu', 'gelu'})
parser.add_argument('--range_flow', type=str, default='logit', choices={'logit', 'softplus'})
parser.add_argument('--log_transform', type=eval, default=False)

# Train params
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--optimizer', type=str, default='adam', choices={'adam', 'adamax'})
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--device', type=str, default='cpu')

# Exp params
parser.add_argument('--rate', type=int, default=1000)

args = parser.parse_args()

train, test = fetch_base_dataset(args)

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators=500, random_state=25)
# Train the model on training data

rf.fit(train[:, 1:], train[:, 0])

pred = rf.predict(test[:, 1:])

pred = np.asarray(pred)
true = test[:, 0]


#print(np.sum(true==1))
#print(true[:,0]==1)
#print(np.sum(pred))
fpr, tpr, thresholds = metrics.roc_curve(true, pred, pos_label=1)
auc = metrics.auc(fpr, tpr)
auc = metrics.roc_auc_score(true, pred)
print(auc)
#yl = np.asarray(yl)
#print(pred.shape, yl.shape)

import matplotlib.pyplot as plt


"""yl = test[:, 0]#test_tensor_y[:, 0]
xl = test[:, 1:]#test_tensor_x


pred = np.asarray(pred)
yl = np.asarray(yl)
xl = np.asarray(xl)
minus = pred - yl
il = np.linspace(0,2500, num=2500)
z1 = (xl[:, 0] == 1)
z0 = (xl[:, 0] == 0)
#il = np.asarray(il)
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
print(npmse(yl, pred))"""