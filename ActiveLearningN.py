import torch
import numpy as np
import gpytorch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import imageio.v2 as imageio
import os
# Part 1: Initial Data
def actualf(x):
    xi = x[:, 0]
    yi = x[:, 1]
    return torch.exp(0.2 * xi) + 2 * torch.cos(xi) * torch.sin(0.5 * yi) + yi - 0.1 * yi * yi
initialN = 20
L = 2 # dimensions
torch.manual_seed(42)
dimCutoffs = torch.tensor([[0,16],[0,10]])
stdNoise = 1
train_x = (dimCutoffs[:, 1] - dimCutoffs[:, 0]) * torch.rand(initialN, L) + dimCutoffs[:, 0]
train_y = actualf(train_x)
train_y += 0.5 * torch.randn_like(train_y)
measurementNoise = torch.ones(initialN)*stdNoise


# 2. Define the Gaussian process model
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.LinearMean(L)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise = measurementNoise)
model = ExactGPModel(train_x, train_y, likelihood)

# 3. Optimize Parameters
optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=0.2)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
prev_loss = float('inf')
loss_change_threshold = 1e-5
for i in range(2000):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y.squeeze())
    loss.backward()
    optimizer.step()
    print("Iteration: ", i, "\t Loss:", loss.item(),"\t Length:",model.covar_module.base_kernel.lengthscale.item())
    if abs(prev_loss - loss.item()) < loss_change_threshold:
        break
    prev_loss = loss.item()

# Manually tweaking parameters
# model.covar_module.base_kernel.lengthscale = .5
# model.likelihood.noise_covar.noise = .002
# 4. Evaluate Model on Test Data and generate uncertainty + predictions
model.eval()
model.likelihood.eval()
x1_min, x1_max = dimCutoffs[0]
x2_min, x2_max = dimCutoffs[1]
test_x1, test_x2 = torch.meshgrid(torch.linspace(x1_min, x1_max, 100), torch.linspace(x2_min, x2_max, 100),indexing='ij')
test_x = torch.stack((test_x1.reshape(-1), test_x2.reshape(-1)), dim=1)
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(test_x),noise=torch.zeros(100**L))
real_y = actualf(test_x).reshape(test_x1.shape)

# 5. Plot
# Demo plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for i in range(initialN):
    ax.errorbar(train_x[i, 0], train_x[i, 1], train_y[i], xerr=0, yerr=0, zerr=measurementNoise[i], fmt='o', ecolor='black',label=None)
# surf = ax.plot_surface(test_x1.numpy(), test_x2.numpy(), real_y.numpy(), cmap='gray', edgecolor='none', alpha=0.7,label='Real')
surf2 = ax.plot_surface(test_x1.numpy(), test_x2.numpy(), observed_pred.mean.reshape(test_x1.shape), cmap='viridis', edgecolor='none', alpha=0.7,label='Predicted')
lowerS = observed_pred.mean - observed_pred.stddev
upperS = observed_pred.mean + observed_pred.stddev
surfL = ax.plot_surface(test_x1.numpy(),test_x2.numpy(),lowerS.reshape(test_x1.shape),cmap='gray',edgecolor='none', alpha=0.7,label='Lower Bound')
surfU = ax.plot_surface(test_x1.numpy(),test_x2.numpy(),upperS.reshape(test_x1.shape),cmap='gray',edgecolor='none', alpha=0.7,label='Upper Bound')
plt.title('Uncertainty of the Trained Model')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x)')
ax.set_title('GPR of f(x)')
# plt.legend()
ax.view_init(elev=30, azim=135)
plt.savefig('3D_SurfPlot.png')
plt.show()

# Part 2: Active learning
activePoints = 50
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    unc = model.likelihood(model(test_x),noise=torch.zeros(100**L))
testedPoints = np.zeros((activePoints,L+1))
for j in range(activePoints):
    # 1. Find point with maximum uncertainty
    new_point = test_x[np.argmax(unc.stddev.numpy())].reshape(1,L)
    new_y = actualf(new_point)
    testedPoints[j,0:L] = new_point.numpy()
    testedPoints[j,L] = new_y.numpy()
    model = model.get_fantasy_model(new_point, torch.tensor([[new_y]]), noise=torch.tensor([stdNoise], dtype=torch.float32))
    # 2. Test New Data on Fantasy model
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        unc = model.likelihood(model(test_x), noise=torch.zeros(100 ** L))
    # 3. Graph
    plt.clf()
    plt.close()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(initialN): # initial data
        ax.errorbar(train_x[i, 0], train_x[i, 1], train_y[i], xerr=0, yerr=0, zerr=measurementNoise[i], fmt='o',
                    ecolor='black', label=None)
    for i in range(j): # active data
        ax.errorbar(testedPoints[i,0],testedPoints[i,1],testedPoints[i,2],xerr=0,yerr=0,zerr=stdNoise,fmt='o',ecolor='red',label=None)

    surf = ax.plot_surface(test_x1.numpy(), test_x2.numpy(), real_y.numpy(), cmap='gray', edgecolor='none', alpha=0.7,label='Real')
    surf2 = ax.plot_surface(test_x1.numpy(), test_x2.numpy(), unc.mean.reshape(test_x1.shape), cmap='viridis',
                            edgecolor='none', alpha=0.7, label='Predicted')
    lowerS = unc.mean - unc.stddev
    upperS = unc.mean + unc.stddev
    # surfL = ax.plot_surface(test_x1.numpy(), test_x2.numpy(), lowerS.reshape(test_x1.shape), cmap='gray',edgecolor='none', alpha=0.7, label='Lower Bound')
    # surfU = ax.plot_surface(test_x1.numpy(), test_x2.numpy(), upperS.reshape(test_x1.shape), cmap='gray',edgecolor='none', alpha=0.7, label='Upper Bound')
    plt.title('Active Learning')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x)')
    ax.set_title('GPR of f(x)')
    ax.view_init(elev=30, azim=135)

    plt.savefig('ActiveUpdatesSurface_'+str(j)+'.png',dpi=300)

# List of filenames of the PNG images
image_filenames = ['ActiveUpdatesSurface_'+str(i)+'.png' for i in range(activePoints)]
def animate_images_as_gif(filenames):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')  # Turn off the axis
    ims = []
    for filename in filenames:
        img = plt.imread(filename)
        im = ax.imshow(img, animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=True)
    ani.save('ActiveSurface.gif', writer='pillow', fps=1, dpi=1000)
animate_images_as_gif(image_filenames)
for filename in image_filenames:
    os.remove(filename)