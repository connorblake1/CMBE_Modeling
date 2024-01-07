import numpy as np
import pandas as pd
import torch
import gpytorch
import matplotlib.pyplot as plt
import joblib
from scipy.optimize import minimize
import os
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
doProbing = False
retrainInitialModel = True

postfix = ""
cols = ['T_cell', 'time', 'wavelength']
names = ["T_cell Temperature (C)", "Time (s)", "Wavelength (nm)"]
path = 'datagroup_10_05_reframed_no817.csv'
doCalib = True
if doCalib:
    postfix = "_calib"
    path = 'datagroup_10_05_no817_calib.csv'
    cols = ['T_cell', 'time', 'wavelength','443Rc','443Ac','514Rc','514Ac','689Rc','689Ac','781Rc','781Ac']
    names = ["T_cell Temperature (C)", "Time (s)", "Wavelength (nm)",'443Rc','443Ac','514Rc','514Ac','689Rc','689Ac','781Rc','781Ac']
L = len(cols)
newnoiseA = .002
newnoiseR = .002
# 1. DEFINE "ACTUAL" GPR based on a ton of data (loaded in)
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()  # LinearMean(L)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=L))
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
# df = pd.read_csv('https://drive.google.com/uc?export=download&id=19IgGl230AbBuaH8q9wOxZ2R4OtRfHb7c')  # no 817
df = pd.read_csv(path)

xtrainA = torch.load('xtrainA'+postfix+'.pt')  # what the full loadedA model was trained on (scaled)
ytrainA = torch.load('ytrainA'+postfix+'.pt')  # what the full loadedA model was trained on (scaled)
xtrainR = torch.load('xtrainR'+postfix+'.pt')  # what the full loadedR model was trained on (scaled)
ytrainR = torch.load('ytrainR'+postfix+'.pt')  # what the full loadedR model was trained on (scaled)
xtestA = torch.load('xtestA'+postfix+'.pt')  # splittest data for loadedA (scaled)
ytestA = torch.load('ytestA'+postfix+'.pt')  # splittest data for loadedA (scaled)
xtestR = torch.load('xtestR'+postfix+'.pt')  # splittest data for loadedR (scaled)
ytestR = torch.load('ytestR'+postfix+'.pt')  # splittest data for loadedR (scaled)
scaler = joblib.load('scaler'+postfix+'.pkl')
likelihood_A = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=newnoiseA*torch.ones(len(xtrainA)))
likelihood_R = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=newnoiseR*torch.ones(len(xtrainR)))
loadedA = ExactGPModel(xtrainA, ytrainA, likelihood_A)
loadedR = ExactGPModel(xtrainR, ytrainR, likelihood_R)
loadedA.load_state_dict(torch.load('model_A'+postfix+'.pth'))
loadedR.load_state_dict(torch.load('model_R'+postfix+'.pth'))
loadedA.eval()
loadedR.eval()
loadedA.likelihood.eval()
loadedR.likelihood.eval()
likelihood_A.eval()
likelihood_R.eval()
def probeModel(ni,T_cellfix,t_fix,wav_fix,lR,mR,lA,mA,filename,pts=None):
    """Make graphs of model with sweep ranges.
        ni: 0 = sweep temp, 1 = time, 2 = wavelength
        """
    if not os.path.exists(filename):
        os.makedirs(filename)
    shortnames = cols
    sweep_values_0 = torch.arange(840, 910, 1)
    sweep_values_1 = torch.arange(1000,6000,10)
    sweep_values_2 = torch.arange(440,800,10)
    sweep_values = [sweep_values_0,sweep_values_1,sweep_values_2]
    test_points = torch.zeros((len(sweep_values[ni]),L))
    for i in range(len(test_points)):
        if ni == 0:
            test_points[i] = torch.tensor([sweep_values[ni][i],t_fix,wav_fix])
        elif ni == 1:
            test_points[i] = torch.tensor([T_cellfix, sweep_values[ni][i], wav_fix])
        elif ni == 2:
            test_points[i] = torch.tensor([T_cellfix, t_fix, sweep_values[ni][i]])
    test_points = torch.from_numpy(scaler.transform(test_points))
    with torch.no_grad():
        predictionsRT = lR(mR(test_points),noise=newnoiseR*torch.ones(len(test_points)))
        predictionsA = lA(mA(test_points),noise=newnoiseA*torch.ones(len(test_points)))
    plt.figure()
    plt.plot(sweep_values[ni].numpy(), predictionsRT.mean.numpy(), color='red',label="RT")
    plt.plot(sweep_values[ni].numpy(), predictionsA.mean.numpy(), color='orange',label="A")
    if pts is not None:
        plt.scatter(pts[0],pts[1],label='o')
    plt.fill_between(
        sweep_values[ni].numpy(), predictionsRT.mean.numpy()+predictionsRT.stddev.numpy(),predictionsRT.mean.numpy()-predictionsRT.stddev.numpy(),
        alpha=0.2, color='blue')
    plt.fill_between(
        sweep_values[ni].numpy(), predictionsA.mean.numpy()+predictionsA.stddev.numpy(),predictionsA.mean.numpy()-predictionsA.stddev.numpy(),
        alpha=0.2, color='green')
    plt.xlabel(names[ni])
    plt.legend()
    plt.title(shortnames[ni] + " Sweep")
    # Save the plot in the parameter_sweep_LR_0.01 folder
    plt.savefig(filename+"/"+shortnames[ni]+'_'+str(T_cellfix)+'_'+str(t_fix)+"_"+str(wav_fix)+'.png')
    plt.close()
# sweep mode (temp, time, wav), temp, time, wav, r0
# Generate Graphs to look at loaded model
if doProbing:
    fname = "Loaded" + str(newnoiseR) + postfix
    RT_6000_689 = df[(df['time'] == 6000) & (df['wavelength'] == 689)][['T_cell', 'RT', 'A']].to_numpy()
    probeModel(0,880,6000,689,likelihood_R,loadedR,likelihood_A,loadedA,fname,pts=(RT_6000_689[:,0],RT_6000_689[:,1]))# temp
    RT_6000_443 = df[(df['time'] == 6000) & (df['wavelength'] == 443)][['T_cell', 'RT', 'A']].to_numpy()
    probeModel(0,800,6000,443,likelihood_R,loadedR,likelihood_A,loadedA,fname,pts=(RT_6000_443[:,0],RT_6000_443[:,1]))
    RT_6000_781 = df[(df['time'] == 6000) & (df['wavelength'] == 781)][['T_cell', 'RT', 'A']].to_numpy()
    probeModel(0,800,6000,781,likelihood_R,loadedR,likelihood_A,loadedA,fname,pts=(RT_6000_781[:,0],RT_6000_781[:,1]))
    RT_880_781 = df[(df['T_cell'] == 880) & (df['wavelength'] == 689)][['time', 'RT', 'A']].to_numpy()
    probeModel(1,880,6000,689,likelihood_R,loadedR,likelihood_A,loadedA,fname,pts=(RT_880_781[:,0],RT_880_781[:,1])) # time
    RT_895_689 = df[(df['T_cell'] == 895) & (df['wavelength'] == 689)][['time', 'RT', 'A']].to_numpy()
    probeModel(1, 895, 6000, 689, likelihood_R, loadedR, likelihood_A, loadedA, fname,pts=(RT_895_689[:, 0], RT_895_689[:, 1]))
    RT_880_6000 = df[(df['T_cell'] == 880) & (df['time'] == 6000)][['wavelength', 'RT', 'A']].to_numpy()
    probeModel(2,880,6000,689,likelihood_R,loadedR,likelihood_A,loadedA,fname,pts=(RT_880_6000[:,0],RT_880_6000[:,1])) # wavelength
def ModelCall(x,iL,iM):
    with torch.no_grad():
        observed = iL(iM(x), noise=newnoiseR*torch.ones(len(x)))
    return observed.mean

# Priorities
# TODO fix make uncertainty not constant - understand what is happening
    # probably because wavelength too sparse and jacking up all uncertainty idk
# TODO calibration layer with new dataset
# TODO active learn the input samples too (fancy interp)
# TODO be able to quantify A* relative to other values of A at that RT
# TODO LSTM interpolation
# TODO LSTM stochastic optimal control
# TODO LSTM model predictive control
# TODO add in noise in "realfake" model
# 2. Set up a model on initialN points
unique_samples = df['substrate'].unique()
num_samples = int(0.4 * len(unique_samples))
random_subset = np.random.choice(unique_samples, size=num_samples, replace=False)
indices = df[df['substrate'].isin(random_subset)].index
initialN = len(indices)
print("Training Selection")
print(unique_samples)
print(random_subset)
print(initialN)

train_x = torch.tensor(df.loc[indices,cols].values,dtype=torch.float32).reshape(-1, L)
train_A = torch.tensor(df.loc[indices,['A']].values, dtype=torch.float32).reshape(initialN)
train_RT = torch.tensor(df.loc[indices,['RT']].values, dtype=torch.float32).reshape(initialN)
train_x = torch.from_numpy(scaler.transform(train_x))
if retrainInitialModel:
    modelRT_likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=torch.ones(initialN)*newnoiseR)
    modelRT = ExactGPModel(train_x, train_RT, modelRT_likelihood).double()
    modelA_likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=torch.ones(initialN)*newnoiseA)
    modelA = ExactGPModel(train_x, train_A, modelA_likelihood).double()
    # Optimize miniature models
    optimizerRT = torch.optim.Adam([{'params': modelRT.parameters()}], lr=0.1)
    mllRT = gpytorch.mlls.ExactMarginalLogLikelihood(modelRT_likelihood, modelRT)
    prev_loss = float('inf')
    loss_change_threshold = 1e-4
    for i in range(2000):
        optimizerRT.zero_grad()
        outputRT = modelRT(train_x)
        lossRT = -mllRT(outputRT, train_RT.squeeze())
        lossRT.backward()
        optimizerRT.step()
        print("Iteration: ", i, "\t Loss:", lossRT.item(),"\t Length:")
        print(modelRT.covar_module.base_kernel.lengthscale)
        if abs(prev_loss - lossRT.item()) < loss_change_threshold:
            break
        prev_loss = lossRT.item()
    optimizerA = torch.optim.Adam([{'params': modelA.parameters()}], lr=0.1)
    mllA = gpytorch.mlls.ExactMarginalLogLikelihood(modelA_likelihood, modelA)
    prev_loss = float('inf')
    for i in range(2000):
        optimizerA.zero_grad()
        outputA = modelA(train_x)
        lossA = -mllRT(outputA, train_A.squeeze())
        lossA.backward()
        optimizerA.step()
        print("Iteration: ", i, "\t Loss:", lossA.item(),"\t Length:")
        print(modelA.covar_module.base_kernel.lengthscale)
        if abs(prev_loss - lossA.item()) < loss_change_threshold:
            break
        prev_loss = lossA.item()
    modelRT.eval()
    modelRT.likelihood.eval()
    modelA.eval()
    modelA.likelihood.eval()
    # Save models
    torch.save(modelRT.state_dict(),'initialN_RT'+postfix+'.pth')
    torch.save(modelA.state_dict(), 'initialN_A'+postfix+'.pth')
    # Validate models with some predictions
    if doProbing:
        probeModel(0, 880, 6000, 689, modelRT_likelihood, modelRT, modelA_likelihood, modelA, "initialN")  # temp
        probeModel(0, 800, 6000, 443, modelRT_likelihood, modelRT, modelA_likelihood, modelA, "initialN")
        probeModel(0, 800, 6000, 781, modelRT_likelihood, modelRT, modelA_likelihood, modelA, "initialN")
        probeModel(1, 880, 6000, 689, modelRT_likelihood, modelRT, modelA_likelihood, modelA, "initialN")  # time
        probeModel(2, 880, 6000, 689, modelRT_likelihood, modelRT, modelA_likelihood, modelA, "initialN")  # wavelength
    with torch.no_grad():
        # y_preds_A = likelihood_A(loadedA(xtestA),noise=newnoiseA*torch.ones(len(xtestA)))
        y_splittest_R = likelihood_R(loadedR(xtestR), noise=newnoiseR * torch.ones(len(xtestR)))
        y_splittest_NR = modelRT_likelihood(modelRT(xtestR), noise=newnoiseR * torch.ones(len(xtestR)))
        y_boot_NR = modelRT_likelihood(modelRT(train_x), noise=newnoiseR * torch.ones(len(train_x)))
    print('Mean abs error (actual vs loadedR xtestR predictions) RT:', np.mean(np.abs(ytestR.detach().numpy() - y_splittest_R.mean.detach().numpy())), " (Full N = ", len(xtrainR),")")
    print('Mean abs error (actual vs initialN xtestR predictions) RT:',
          np.mean(np.abs(ytestR.detach().numpy() - y_splittest_NR.mean.detach().numpy())), " (initialN = ", initialN,")")
    print('Mean abs error (loadedR xtestR predictions vs initialN xtestR predictions) RT:',
          np.mean(np.abs(y_splittest_R.mean.detach().numpy() - y_splittest_NR.mean.detach().numpy())))
    print('Mean abs error (loadedR values on initialN points vs new model on initialN points) RT:',
          np.mean(np.abs(train_RT.detach().numpy() - y_boot_NR.mean.detach().numpy())))
else:  # load initialN models
    modelRT_likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=torch.ones(len(train_x))*newnoiseR)
    modelA_likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=torch.ones(len(train_x))*newnoiseA)
    modelRT = ExactGPModel(train_x, train_RT, modelRT_likelihood).double()
    modelA = ExactGPModel(train_x, train_A, modelA_likelihood).double()
    modelRT.load_state_dict(torch.load('initialN_RT'+postfix+'.pth'))
    modelA.load_state_dict(torch.load('initialN_A'+postfix+'.pth'))
    modelA.eval()
    modelRT.eval()
    modelA.likelihood.eval()
    modelRT.likelihood.eval()
    modelA_likelihood.eval()
    modelRT_likelihood.eval()
exit()
# 3. Algo
# a) find argmin of the loss function of the GPRs (modelA, modelRT) (lowest = optimal inputs for these GPRs and loss fxns) (will need to do gradient descent on)
# b) execute growth (call loadedA, loadedR)
# c) use fantasy_model to update learned GPRs
# d) if below some loss threshold, move on to new loss function, else stay and keep argmining with new data
def loss(RT,A,tRT,cRT,cA): # convex wrt RT, A
    L = 0
    for i,R in enumerate(RT):
        L += cRT[i]*(R-tRT[i])**2
        L += cA[i]*A[i]*A[i]
    return L

# Ideal Growths:
sampleNum = 1
wavs = [443, 514, 689, 781]
targets = [(0,.85,1),(0,.44,1),(0,.23,1),(1,.56,1),(0,.57,1)] # tuples with wavelength wavs, target RT, penalty (absoprtion square weighted x missing RT value)
lossCutoff = .0001
maxAttempts = 10
print("Starting Growth (" + str(len(targets)) + " targets)")
for target in targets:
    cRT = [0]*len(wavs)
    tRT = [0]*len(wavs)
    cA = [0]*len(wavs)
    cRT[target[0]] = 100
    tRT[target[0]] = target[1]
    cA[target[0]] = target[2]
    def CallLoss(x):
        T_c, t = x
        RT = []
        A = []
        for wav in wavs:
            tpoint = torch.tensor([T_c, t, wav]).reshape(1,-1)
            tpoint = torch.from_numpy(scaler.transform(tpoint))
            with torch.no_grad():
                tpoint_NR = modelRT_likelihood(modelRT(tpoint), noise=newnoiseR * torch.ones(1))
                tpoint_NA = modelA_likelihood(modelA(tpoint), noise=newnoiseA * torch.ones(1))
            RT.append(tpoint_NR.mean)
            A.append(tpoint_NA.mean)
        return loss(RT,A,tRT,cRT,cA)
    lossVal = float('inf')
    targetAttempts = 1
    while targetAttempts <= maxAttempts:
        print("Sample Number:" + str(sampleNum))
        sampleNum += 1
        print("\tTarget: Wav = " + str(wavs[target[0]]) + ", RT = " + str(target[1]))
        print("\tFinding optimal...")
        bounds = [(840,910), (0,6000)]
        initial_guess = [870,3000]
        result = minimize(fun=CallLoss, x0=initial_guess, bounds=bounds,tol=1e-9,jac="3-point",method='L-BFGS-B')
        # print("\t"+str(result.message))
        conditionT = result.x[0]
        conditiont = result.x[1]
        predLoss = CallLoss([conditionT,conditiont]).item()
        conditionTensor = torch.from_numpy(scaler.transform(torch.tensor([conditionT,conditiont,wavs[target[0]]]).reshape(1,-1))).reshape(1,-1)
        RT_model = ModelCall(conditionTensor,modelRT_likelihood,modelRT).item()
        A_model = ModelCall(conditionTensor,modelA_likelihood,modelA).item()
        print("\tConditions Found: Temp (C)=" + str(np.round(conditionT,2)) + ", Time (s)="+str(np.round(conditiont,2))+"\t\tExpected Loss: " + str(np.round(predLoss,3)))
        print("\tExpected Values:\tRT="+str(np.round(RT_model,3))+", A="+str(np.round(A_model,3)))

        print("\tGrowth...")
        RT_grow = ModelCall(conditionTensor,likelihood_R,loadedR).item()
        A_grow = ModelCall(conditionTensor,likelihood_A,loadedA).item()
        RT = [0]*len(wavs)
        A = [0]*len(wavs)
        RT[target[0]] = RT_grow
        A[target[0]] = A_grow
        lossVal = loss(RT, A, tRT, cRT, cA)
        convergenceLoss = (RT_grow-RT_model)**2
        print("\tGrowth Yields:\t\tRT=" + str(np.round(RT_grow,3)) + ", A=" + str(np.round(A_grow,3)) + "\t\tActual Loss: " + str(np.round(lossVal,3)) + "\t\tConvergence Loss: " + str(np.round(convergenceLoss,5)))

        print("\tUpdating GPRs...")
        point_weight = 6
        ct2 = torch.cat([conditionTensor] * point_weight, dim=0)
        modelRT = modelRT.get_fantasy_model(ct2, torch.tensor([point_weight*[RT_grow]]),noise=torch.tensor(point_weight*[0]))
        modelA = modelA.get_fantasy_model(ct2,torch.tensor([point_weight*[A_grow]]),noise=torch.tensor(point_weight*[0]))

        if convergenceLoss < lossCutoff:
            print("\tThis growth (Temp (C)=" + str(np.round(conditionT,2)) + ", Time (s)="+str(np.round(conditiont,2))+") PASSED. On to the next target!")
            break
        else:
            print("\tThis growth FAILED. Retrying (" + str(targetAttempts)+"/"+str(maxAttempts)+")")
            targetAttempts += 1
