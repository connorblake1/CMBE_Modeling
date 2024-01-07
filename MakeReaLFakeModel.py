import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import KFold
from sklearn.gaussian_process.kernels import PairwiseKernel
import gpytorch
import torch
import joblib
import os
# Regression
doCalib = True
postfix = ""
cols = ['T_cell', 'time', 'wavelength']
# link: 'https://drive.google.com/uc?export=download&id=19IgGl230AbBuaH8q9wOxZ2R4OtRfHb7c' (no 817)
path = 'file:///C:/Users/theco/PycharmProjects/CMBE_Parsing/datagroup_10_05_reframed_no817.csv'
if doCalib:
    postfix = "_calib"
    cols = ['T_cell', 'time', 'wavelength','443Rc','443Ac','514Rc','514Ac','689Rc','689Ac','781Rc','781Ac']
    path = 'file:///C:/Users/theco/PycharmProjects/CMBE_Parsing/datagroup_10_05_no817_calib.csv'
df = pd.read_csv(path)
# Split the data into training and test sets
DIMS = len(cols)
X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(df[cols], df['A'].to_numpy(), test_size=0.25)
X_train_R, X_test_R, y_train_R, y_test_R = train_test_split(df[cols], df['RT'].to_numpy(), test_size=0.25)
scaler = StandardScaler()
scaler.fit(X_train_A)
print(scaler.mean_)
print(scaler.scale_)
print(scaler.var_)
joblib.dump(scaler, 'scaler'+postfix+'.pkl')
X_train_A = torch.from_numpy(scaler.transform(X_train_A))
X_train_R = torch.from_numpy(scaler.transform(X_train_R))
X_test_A = torch.from_numpy(scaler.transform(X_test_A))
X_test_R = torch.from_numpy(scaler.transform(X_test_R))
y_train_A, y_test_A = torch.from_numpy(y_train_A), torch.from_numpy(y_test_A)
y_train_R, y_test_R = torch.from_numpy(y_train_R), torch.from_numpy(y_test_R)
# X_train_R = torch.tensor(X_train_R.values, dtype=torch.float32).reshape(-1,4)
# y_train_R = torch.tensor(y_train_R, dtype=torch.float32).reshape(-1)
# X_train_A = torch.tensor(X_train_A.values, dtype=torch.float32).reshape(-1,4)
# y_train_A = torch.tensor(y_train_A,dtype=torch.float32).reshape(-1)
# X_test_R = torch.tensor(X_test_R.values, dtype=torch.float32).reshape(-1,4)
# y_test_R = torch.tensor(y_test_R, dtype=torch.float32).reshape(-1)
# X_test_A = torch.tensor(X_test_A.values, dtype=torch.float32).reshape(-1,4)
# y_test_A = torch.tensor(y_test_A,dtype=torch.float32).reshape(-1)
# X_train_A = torch.from_numpy(scaler.inverse_transform(X_train_A.numpy()))
# X_train_R = torch.from_numpy(scaler.inverse_transform(X_train_R.numpy()))
# X_test_A = torch.from_numpy(scaler.inverse_transform(X_test_A.numpy()))
# X_test_R = torch.from_numpy(scaler.inverse_transform(X_test_R.numpy()))
torch.save(X_train_A, 'xtrainA'+postfix+'.pt')
torch.save(y_train_A, 'ytrainA'+postfix+'.pt')
torch.save(X_train_R, 'xtrainR'+postfix+'.pt')
torch.save(y_train_R, 'ytrainR'+postfix+'.pt')
torch.save(X_test_A, 'xtestA'+postfix+'.pt')
torch.save(y_test_A, 'ytestA'+postfix+'.pt')
torch.save(X_test_R, 'xtestR'+postfix+'.pt')
torch.save(y_test_R, 'ytestR'+postfix+'.pt')

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()  # LinearMean(DIMS)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=DIMS))
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
noiseA = .002
noiseR = .002
likelihood_A = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noiseA*torch.ones(len(X_train_A)))
likelihood_R = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noiseR*torch.ones(len(X_train_R)))
model_A = ExactGPModel(X_train_A, y_train_A, likelihood_A)
model_R = ExactGPModel(X_train_R, y_train_R, likelihood_R)  # NOTE: this can take in dataframes and treat them like tensors even if they are NOT
model_A.train()
model_R.train()
likelihood_A.train()
likelihood_R.train()
# model_A.eval()
# model_R.eval()
# likelihood_A.eval()
# likelihood_R.eval()
print("A Optimization")
optimizer_A = torch.optim.Adam([{'params': model_A.parameters()}], lr=0.1)
mll_A = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_A, model_A)
prev_loss = float('inf')
loss_change_threshold = 1e-5
for i in range(2000):
    optimizer_A.zero_grad()
    output_A = model_A(X_train_A)
    loss_A = -mll_A(output_A, y_train_A.squeeze())
    loss_A.backward()
    optimizer_A.step()
    print("Iteration: ", i, "\t Loss:", loss_A.item(), "\t Length:")
    print(model_A.covar_module.base_kernel.lengthscale)
    # print(model_A.likelihood.noise_covar.noise)
    if abs(prev_loss - loss_A.item()) < loss_change_threshold:
        break
    prev_loss = loss_A.item()
torch.save(model_A.state_dict(), 'model_A'+postfix+'.pth')
print("R Optimization")
optimizer_R = torch.optim.Adam([{'params': model_R.parameters()}], lr=0.1)
mll_R = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_R, model_R)
prev_loss = float('inf')
for i in range(2000):
    optimizer_R.zero_grad()
    output_R = model_R(X_train_R)
    loss_R = -mll_R(output_R, y_train_R.squeeze())
    loss_R.backward()
    optimizer_R.step()
    print("Iteration: ", i, "\t Loss:", loss_R.item(),"\t Length:")
    print(model_R.covar_module.base_kernel.lengthscale)
    if abs(prev_loss - loss_R.item()) < loss_change_threshold:
        break
    prev_loss = loss_R.item()
torch.save(model_R.state_dict(), 'model_R'+postfix+'.pth')


