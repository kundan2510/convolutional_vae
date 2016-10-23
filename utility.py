from theano.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams(seed=234)

import theano
import theano.tensor as T

import numpy as np

import scipy

def floatX(a):
	if theano.config.floatX == 'float32':
		return np.float32(a)
	elif theano.config.floatX == 'float16':
		return np.float32(a)
	else:
		raise NotImplementedError("{} type-casting not implemented".format(theano.config.floatX))

def uniform(stdev, size):
    """uniform distribution with the given stdev and size"""
    return np.random.uniform(
        low=-stdev * np.sqrt(3),
        high=stdev * np.sqrt(3),
        size=size
    ).astype(theano.config.floatX)

def binarize(X):
	un = np.random.uniform(low=0., high=1., size = X.shape)
	X[X > un] = 1.
	X[X <= un] = 0.
	return X 

def plot_100_figure(images, output_name, num_channels = 1):
	HEIGHT, WIDTH = images.shape[1], images.shape[2]
	if num_channels == 1:
		images = images.reshape((10,10,HEIGHT,WIDTH))
		# rowx, rowy, height, width -> rowy, height, rowx, width
		images = images.transpose(1,2,0,3)
		images = images.reshape((10*28, 10*28))
		scipy.misc.toimage(images, cmin=0.0, cmax=1.0).save(output_name)
	elif num_channels == 3:
		images = images.reshape((10,10,HEIGHT,WIDTH,3))
		images = images.transpose(1,2,0,3,4)
		images = images.reshape((10*HEIGHT, 10*WIDTH, 3))
		scipy.misc.toimage(images).save(output_name)
	else:
		raise Exception("You should not be here!! Only 1 or 3 channels allowed for images!!")

def KL_with_standard_gaussian(mu, log_square_sigma):
	# return (
	# 		floatX(-0.5) * (
	# 				floatX(1.) + (floatX(2.) * log_sigma) - ( mu**2) - \
	# 				T.exp(floatX(2.) * log_sigma)
	# 			)
	# 		).mean(axis=0).sum()
	return -0.5 * T.sum(1 + log_square_sigma - T.square(mu) - T.exp(log_square_sigma), axis=1)


def gaussian_sampler(mu, log_square_sigma):
	eps = T.cast(srng.normal(size=mu.shape), theano.config.floatX)
	return mu + (eps*T.exp(0.5*log_square_sigma))

