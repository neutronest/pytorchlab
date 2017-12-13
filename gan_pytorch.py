import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
from torch.autograd import Variable
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset


import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pdb

def show_images(images):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))
    return 

def preprocess_img(x):
    return 2 * x - 1.0

def deprocess_img(x):
    return (x + 1.0) / 2.0

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def count_params(model):
    """Count the number of parameters in the current TensorFlow graph """
    param_count = np.sum([np.prod(p.size()) for p in model.parameters()])
    return param_count

#answers = np.load('gan-checks-tf.npz')

class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset. 
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples

# const variable
dtype = torch.FloatTensor
NUM_TRAIN = 50000
NUM_VAL = 5000

NOISE_DIM = 96
batch_size = 128
answers = np.load("gan-checks-tf.npz")
mnist_train = dset.MNIST('./cs231n/datasets/MNIST_data', train=True, download=True,
                        transform=T.ToTensor())
loader_train = DataLoader(mnist_train, batch_size=batch_size,
                        sampler=ChunkSampler(NUM_TRAIN, 0))

mnist_val = dset.MNIST('./cs231n/datasets/MNIST_data', train=True, download=True,
                        transform=T.ToTensor())
loader_val = DataLoader(mnist_val, batch_size=batch_size,
                        sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))


imgs = loader_train.__iter__().next()[0].view(batch_size, 784).numpy().squeeze()

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

show_images(imgs)



def sample_noise(batch_size, dim):
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - dim: Integer giving the dimension of noise to generate.
    
    Output:
    - A PyTorch Tensor of shape (batch_size, dim) containing uniform
      random noise in the range (-1, 1).
    """
    return torch.rand((batch_size, dim))*2-1
    pass

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
    
class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """
    def __init__(self, N=-1, C=128, H=7, W=7):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W
    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)

def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform(m.weight.data)


###############################
### Model Structure
###############################



def discriminator():
    dis_model = nn.Sequential(

        Flatten(),
        nn.Linear(784, 256),
        nn.LeakyReLU(0.01),

        nn.Linear(256, 256),
        nn.LeakyReLU(0.01),
        nn.Linear(256, 1)
    )
    return dis_model

def generator(noise_dim=NOISE_DIM):

    gen_model = nn.Sequential(
 
        nn.Linear(noise_dim, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 784),
        nn.Tanh()
    )

    return gen_model

def deepcnn_discriminator(batch_size=128):

    d_model = nn.Sequential(
        # input (N, C, W, H): batch, 3, 28, 28
        Unflatten(batch_size, 1, 28, 28),
        nn.Conv2d(1, 32, kernel_size=5, stride=1), # (N, 32, 24, 24)
        nn.LeakyReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2), # (N, 32, 12, 12)
        nn.Conv2d(32, 64, kernel_size=5, stride=1), # (N, 64, 8, 8)
        nn.LeakyReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2), # (N, 64, 4, 4)
        Flatten(),
        nn.Linear(1024, 1024),
        nn.LeakyReLU(inplace=True),
        nn.Linear(1024, 1)
    )
    return d_model

def deepcnn_generator(noise_dim=10, batch_size=128):

    g_model = nn.Sequential(

        nn.Linear(noise_dim, 1024),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(num_features=1024),
        nn.Linear(1024, 7 * 7 * 256),
        nn.ReLU(inplace=True),  # o_dim = N, 1, 112, 112
        nn.BatchNorm1d(num_features=7*7*256),
        Unflatten(batch_size, 1, 112, 112),
        nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1), # N, 64, 56, 56
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=64),
        nn.Conv2d(64, 1, kernel_size=4, stride=2, padding=1), # N, 1, 28, 28
        nn.Tanh(),
        Flatten()
    )

    return g_model

#####################################
### Loss Definition
#####################################


def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.

    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

    Inputs:
    - input: PyTorch Variable of shape (N, ) giving scores.
    - target: PyTorch Variable of shape (N,) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Variable containing the mean BCE loss over the minibatch of input data.
    """
    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()

def discriminate_loss(logits_real, logits_fake):

    """
    Computes the discriminator loss described above.
    
    Inputs:
    - logits_real: PyTorch Variable of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Variable of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Variable containing (scalar) the loss for the discriminator.
    """ 
    dtype = torch.FloatTensor
    batch_size = logits_real.data.size
    y_true_labels = Variable(torch.ones(logits_real.data.size())).type(dtype)
    y_false_labels = Variable(torch.zeros(logits_fake.data.size())).type(dtype)
    loss = bce_loss(logits_real, y_true_labels)
    loss = loss + bce_loss(logits_fake, y_false_labels)
    return loss

def generate_loss(logits_fake):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Variable of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Variable containing the (scalar) loss for the generator.
    """
    return - torch.mean(logits_fake)


def ls_discriminate_loss(score_real, score_fake):

    """
    Compute the Least-Squares GAN loss for the discriminator.
    
    Inputs:
    - scores_real: PyTorch Variable of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Variable of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Variable containing the loss.
    """
    loss = 0.5 * (torch.mean((score_real - 1) * (score_real - 1)) + torch.mean(score_fake * score_fake) )

    return loss

def ls_generate_loss(score_fake):


    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Variable of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Variable containing the (scalar) loss for the generator.
    """
    loss = 0.5 * torch.mean((score_fake - 1) * (score_fake - 1))
    return loss




def get_optimizer(model):
    """
    Construct and return an Adam optimizer for the model with learning rate 1e-3,
    beta1=0.5, and beta2=0.999.
    
    Input:
    - model: A PyTorch model that we want to optimize.
    
    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
    """
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas = [0.5, 0.999])
    return optimizer

def run_gan(d_model, g_model, 
    d_solver, g_solver, 
    d_loss_fn, g_loss_fn, 
    show_every=250, batch_size=128,
    noise_size=96, num_epoches=10):

    """
    Train a GAN!
    Inputs:
    - d_model, g_model: PyTorch models for the discriminator and generator
    - d_solver, g_solver: torch.optim Optimizers to use for training the
      discriminator and generator.
    - d_loss_fn, g_loss_fn: Functions to use for computing the generator and
      discriminator loss, respectively.
    - show_every: Show samples after every show_every iterations.
    - batch_size: Batch size to use for training.
    - noise_size: Dimension of the noise to use as input to the generator.
    - num_epochs: Number of epochs over the training dataset to use for training.
    """
    iter_cnt = 0
    for epoch in range(num_epoches):
        for x, _ in loader_train:
            if len(x) != batch_size:
                continue
            d_solver.zero_grad()
            real_data = Variable(x).type(dtype)
            #print ("real_data's size: {}".format(real_data.data.size()))
            logits_real = d_model(2 * (real_data - 0.5)).type(dtype)

            g_fate_seed = Variable(sample_noise(batch_size, noise_size)).type(dtype)
            fake_images = g_model(g_fate_seed).detach() # split a new variable
            logits_fake = d_model(fake_images.view(batch_size, 1, 28, 28))

            d_total_loss = d_loss_fn(logits_real, logits_fake)
            d_total_loss.backward()
            d_solver.step()

            g_solver.zero_grad()
            gen_g_fake_seed = Variable(sample_noise(batch_size, noise_size)).type(dtype)
            gen_fake_images = g_model(gen_g_fake_seed)
            gen_logits_fake = d_model(gen_fake_images.view(batch_size, 1, 28, 28))
            g_loss = g_loss_fn(gen_logits_fake)
            g_loss.backward()
            g_solver.step()

            # check if print
            if (iter_cnt % show_every == 0):
                print("Iter: {}, d: {:.4}, g: {:.4}".format(iter_cnt, 
                    d_total_loss.data[0],
                    g_loss.data[0]))
                imgs_numpy = gen_fake_images.data.cpu().numpy()
                show_images(imgs_numpy[0:16])
                #plt.show()
                plt.savefig("./figures/naive_model_%s.png"%(str(iter_cnt)) )
                print()
            iter_cnt += 1
    return


def test_sample_noise():
    batch_size = 3
    dim = 4
    torch.manual_seed(231)
    z = sample_noise(batch_size, dim)
    print (z)
    np_z = z.cpu().numpy()
    assert np_z.shape == (batch_size, dim)
    assert torch.is_tensor(z)
    assert np.all(np_z >= -1.0) and np.all(np_z <= 1.0)
    assert np.any(np_z < 0.0) and np.any(np_z > 0.0)
    print('All tests passed!')

def test_1():
    """
    Test the d model size checking
    """
    print("[Test d_model structure: naive...]")
    dtype = torch.FloatTensor
    dis_model = discriminator()
    #gen_model = generator(4)
    #print (count_params(dis_model))
    #print (count_params(gen_model))
    if count_params(dis_model) != 267009:
        print("model size error!")
    else:
        print("[Ok...]")
    return

def test_2():
    """
    Test the g model size checking
    """
    print("[Test g_model structure naive...]")
    dtype = torch.FloatTensor
    g_model = generator(4)
    if count_params(g_model) != 1858320:
        print("model size error!")
    else:
        print("[Ok...]")
    return

def test_3():
    """
    """
    print("[Test g naive model's loss fn...]")
    dtype = torch.FloatTensor
    logits_fake = Variable(torch.Tensor(answers["logits_fake"])).type(dtype)
    g_loss_true = answers["g_loss_true"]
    g_loss = generate_loss(logits_fake).data.cpu().numpy()
    print ("g_loss: %g", g_loss)
    print ("g_loss_true: %g", g_loss_true)
    print ("maximum loss in g_loss; %g"%(rel_error(g_loss_true, g_loss)))
    print ("[Ok...all]")


    return

def test_4():
    print("[Test d naive model's loss fn...]")
    dtype = torch.FloatTensor
    logits_real = Variable(torch.Tensor(answers["logits_real"])).type(dtype)
    logits_fake = Variable(torch.Tensor(answers["logits_fake"])).type(dtype)
    d_loss = discriminate_loss(logits_real, logits_fake)
    print ("d_loss: %g", d_loss.data.cpu().numpy())
    print ("d_loss_true: %g", answers["d_loss_true"])
    print ("[Ok...]")
    return

def test_5():
    print ("[Test Train first GAN model!]")
    
    dtype = torch.FloatTensor
    #d_model = discriminator().type(dtype)
    #g_model = generator().type(dtype)
    dc_d_model = deepcnn_discriminator(batch_size).type(dtype)
    dc_g_model = deepcnn_generator(NOISE_DIM).type(dtype)
    d_solver = get_optimizer(dc_d_model)
    g_solver = get_optimizer(dc_g_model)
    d_loss_fn = discriminate_loss
    g_loss_fn = generate_loss
    d_ls_loss_fn = ls_discriminate_loss
    g_ls_loss_fn = ls_generate_loss
    run_gan(dc_d_model, dc_g_model, d_solver, g_solver, d_ls_loss_fn, g_ls_loss_fn)
    
    
    print ("[Ok...]")

if __name__ == "__main__":

    #test_1()
    #test_2()
    #test_3()
    #test_4()
    test_5()
