import torch
import numpy as np
import gpytorch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import imageio.v2 as imageio
import os
# Part 1: Initial Data
def actualf(x):
    return np.exp(.2*x) + 2*np.cos(x)
# 1. Give Training Data
DIMS = 1
initialN = 3
torch.manual_seed(40)
lo = 0
hi = 16
stdNoise = .4
train_x = torch.rand(initialN)*(hi-lo)+lo
train_y = torch.exp(.2*train_x)+2*torch.cos(train_x) # exact
train_y += stdNoise * torch.randn_like(train_y) # noisy
scaler = StandardScaler()
scaler.fit(train_x.reshape(-1,DIMS))
train_x_rescale = torch.from_numpy(scaler.transform(train_x.reshape(-1,DIMS)))
print("Mean ",scaler.mean_)
print("Scales ",scaler.scale_)
print("Variance ",scaler.var_)
# 2. Define the Gaussian process model
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()# LinearMean(1)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=torch.ones(initialN)*stdNoise)
model = ExactGPModel(train_x_rescale, train_y, likelihood)

# 3. Optimize Parameters
optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=0.2)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
prev_loss = float('inf')
loss_change_threshold = 1e-5
for i in range(800):
    optimizer.zero_grad()
    output = model(train_x_rescale)
    loss = -mll(output, train_y.squeeze())
    loss.backward()
    optimizer.step()
    # print("Iteration: ", i, "\t Loss:", loss.item(),"\t Length:",model.covar_module.base_kernel.lengthscale.item(),"\t Noise:",likelihood.noise.item())
    if abs(prev_loss - loss.item()) < loss_change_threshold:
        break
    prev_loss = loss.item()

# Manually tweaking parameters
# model.covar_module.base_kernel.lengthscale = .5
# model.likelihood.noise_covar.noise = .002

# 4. Evaluate Model on Test Data and generate uncertainty + predictions
model.eval()
model.likelihood.eval()
x2len = 1000
test_x2 = torch.linspace(-2, 16, x2len)
test_x2_rescale = torch.from_numpy(scaler.transform(test_x2.reshape(-1,DIMS)))
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(test_x2_rescale),noise=stdNoise*torch.ones(x2len))

# 5. Plot
plt.figure(figsize=(8, 6))
plt.errorbar(train_x,train_y,xerr=None,yerr=torch.ones(initialN)*stdNoise,label="Raw Data",fmt="o",linestyle=None)
plt.plot(test_x2.numpy(), observed_pred.mean.numpy(), 'b', label='Mean Prediction')
plt.plot(test_x2.numpy(), actualf(test_x2.numpy()),'g',label='Real')
plt.fill_between(
    test_x2.numpy(), observed_pred.mean.numpy()+observed_pred.stddev.numpy(), observed_pred.mean.numpy()-observed_pred.stddev.numpy(),alpha=0.2, color='blue')
plt.title('Uncertainty of the Trained Model')
plt.xlabel('X')
plt.ylabel('Prediction')
plt.legend()
plt.savefig('Data_'+str(initialN)+'_Noise'+str(np.round(stdNoise,4))+'.png')
plt.show()
# Part 2: ADD NEW DATAPOINTS 1 at a time
# trainingPoints = 20
# # 1. New data
# new_data_x = torch.linspace(4,8,trainingPoints)
# new_data_y = torch.exp(.2*new_data_x)+2*torch.cos(new_data_x)
# new_data_y += 1 * torch.randn_like(new_data_y)
# for j in range(1,trainingPoints,4):
#     # 2. Build Fantasy Model
#     fantasy_model = model.get_fantasy_model(torch.tensor(new_data_x[0:j]), torch.tensor(new_data_y[0:j]),noise=torch.tensor([stdNoise]*j, dtype=torch.float32))
#     # 3. Test New Data on Fantasy model
#     with torch.no_grad(), gpytorch.settings.fast_pred_var():
#         observed_pred_fant = fantasy_model.likelihood(fantasy_model(test_x2))
#     # 4. Graph
#     plt.clf()
#     plt.close()
#     plt.figure(figsize=(8, 6))
#     plt.errorbar(train_x, train_y, xerr=None, yerr=measurementNoise, label="Raw Data", fmt="o", linestyle=None)
#     plt.errorbar(new_data_x[0:j], new_data_y[0:j], xerr=None, yerr=[stdNoise]*(j), label="New Data", fmt="o", linestyle=None)
#     plt.plot(test_x2.numpy(), observed_pred_fant.mean.numpy(), 'b', label='Mean Prediction')
#     plt.fill_between(
#         test_x2.numpy(), *observed_pred_fant.confidence_region(),
#         alpha=0.2, color='blue')
#     plt.title('Uncertainty of the Fantasy Trained Model')
#     plt.xlabel('X')
#     plt.ylabel('Prediction')
#     plt.legend()
#     plt.savefig('FantasyUpdates_'+str(j)+'.png')

# Part 3: Active learning
activePoints = 10
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    unc = model.likelihood(model(test_x2_rescale),noise=stdNoise*torch.ones(x2len))
testedPoints = np.zeros((10,2))
for j in range(activePoints):
    # 1. Find point with maximum uncertainty
    new_point = test_x2_rescale[np.argmax(unc.stddev.numpy())].item() # needs to be a scipy.minimize in higher dims
    rescale_new_x = torch.from_numpy(scaler.inverse_transform(torch.tensor([new_point]).reshape(1,-1)))
    new_y = actualf(rescale_new_x).item()+stdNoise * torch.randn(1)
    testedPoints[j,0] = rescale_new_x
    testedPoints[j,1] = new_y
    print(j,testedPoints[j,0],new_y)
    fantasy_noise = 0.001 # 0 makes model converge, ANYTHING above it doesn't # TODO figure out why
    model = model.get_fantasy_model(torch.tensor([new_point]), torch.tensor([new_y]), noise=torch.tensor([fantasy_noise])) # This is where everything breaks 1/6
    model.eval()
    # 3. Test New Data on Fantasy model
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        unc = model.likelihood(model(test_x2_rescale),noise=stdNoise*torch.ones(x2len))
    # 4. Graph
    plt.close()
    plt.figure(figsize=(8, 6))
    plt.errorbar(train_x, train_y, xerr=None, yerr=torch.ones(initialN)*stdNoise, label="Raw Data", fmt="o", linestyle=None)
    plt.errorbar(testedPoints[0:j+1,0],testedPoints[0:j+1,1], xerr=None, yerr=[stdNoise]*(j+1), label="New Data", fmt="o",
                 linestyle=None)
    uncmean = unc.mean.numpy()
    uncunc = unc.stddev.numpy()
    nicex2 = test_x2.numpy()
    plt.plot(nicex2.reshape(-1,1), uncmean, 'b', label='Mean Prediction')
    plt.plot(test_x2.numpy(), actualf(test_x2.numpy()), 'g', label='Real')
    lower = uncmean-uncunc
    upper = uncmean+uncunc
    plt.fill_between(
        nicex2.flatten(), lower.flatten(),upper.flatten(),
        alpha=0.2, color='blue')
    plt.title('Active Learning Demo')
    plt.xlabel('X')
    plt.ylabel('Prediction')

    plt.xlim([np.min(nicex2)-.5,np.max(nicex2)+.5])
    plt.ylim([np.min(actualf(nicex2))-9,np.max(actualf(nicex2))+4])
    plt.legend()
    plt.savefig('ActiveUpdates_'+str(j)+'.png')

# List of filenames of the PNG images
image_filenames = ['ActiveUpdates_'+str(i)+'.png' for i in range(activePoints)]
# Read the images and create a GIF
images = []
for filename in image_filenames:
    images.append(imageio.imread(filename))
output_gif_filename = 'ActiveLearning_'+str(initialN)+'_ActualNoise'+str(np.round(stdNoise,4))+'_UpdateNoise'+str(np.round(fantasy_noise,4))+'.gif'
imageio.mimsave(output_gif_filename, images,duration=1000)
for filename in image_filenames:
    os.remove(filename)

