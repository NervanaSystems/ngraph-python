# GAN
# following example code from https://github.com/AYLIEN/gan-intro
# MLP generator and discriminator 
# toy example with 1-D Gaussian data distribution
import numpy as np

class ToyGAN(object):  # analogy to MNIST or PTB classes. bleh on what to name this
    """
    Data loader class for toy GAN 1-D Gaussian example

    number of samples N is artifically specified 
    (unlike loading from a dataset like MNIST or PTB with predefined
     number of examples)

    Arguments:
        N (int): total number of samples to create
        data_mu (float): mean of actual Gaussian data distribution
        data_sigma (float): std dev of actual Gaussian data distribution
        noise_range (float): range in stratified sampling noise input to generator
    """
    def __init__(self, batch_size, num_iter, data_mu=4, data_sigma=0.5, noise_range=8):
        # for this toy example, instead of dataset download info,
        # record parameters of Gaussian data and stratified sampling noise input to generator
        self.batch_size = batch_size
        self.num_iter = num_iter
        self.data_mu = data_mu
        self.data_sigma = data_sigma
        self.noise_range = noise_range

    def data_samples(self, bsz, num_iter):
        ds = np.zeros((num_iter, bsz))
        for i in range(num_iter):
            samples = np.random.normal(self.data_mu, self.data_sigma, bsz)
            ds[i] = np.sort(samples)
        return ds.reshape(-1, 1)

    def noise_samples(self, bsz, num_iter):
        # stratified sampling
        ns = np.zeros((num_iter, bsz))
        for i in range(num_iter):
            ns[i] = np.linspace(-self.noise_range, self.noise_range, bsz) + \
                    np.random.random(bsz) * 0.01
        return ns.reshape(-1, 1)

    def load_data(self):
        # assume reshape is necessary
        # data_samples : total number of examples x feature size = N x 1
        data_samples = self.data_samples(self.batch_size, self.num_iter)
        noise_samples = self.noise_samples(self.batch_size, self.num_iter)

        # format expected by ArrayIterator
        self.train_set = {'data_sample': {'data': data_samples,
                                           'axes': ('batch', 'sample')},  # is this correct? axes are weird
                          'noise_sample': {'data': noise_samples,
                                            'axes': ('batch', 'sample')}}
        return self.train_set
