"""
Convolutional variational auto-encoder for MNIST.
Author: Kundan Kumar

Usage: THEANO_FLAGS='mode=FAST_RUN,device=gpu3,floatX=float32,lib.cnmem=.95' python new_vae.py
"""

import sys
sys.path.append("/u/kumarkun/nn/")

from keras.datasets import mnist
from random import shuffle

import theano
import theano.tensor as T
import lasagne
from theano.compile.nanguardmode import NanGuardMode

import numpy as np

import os
import scipy
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from utility import *

import lib
import lib.ops.conv2d
import lib.ops.deconv2d
import lib.ops.relu
import lib.ops.linear

import argparse
parser = argparse.ArgumentParser()

add_arg = parser.add_argument

add_arg("--eps", default=0.0000001, type=floatX, help= "Epsilon for numerical stability")

args = parser.parse_args()

# lib.ops.relu.relu = lambda x: T.switch(x < floatX(2.), T.switch(x > floatX(0.), x, floatX(0.)), floatX(2.))

EPS = args.eps

BATCH_SIZE = 100

OUT_DIR = "mnist_conv_vae"



if not os.path.isdir(OUT_DIR):
	print "Creating directory {}".format(OUT_DIR)
	os.makedirs(OUT_DIR)
	print "Created {}".format(OUT_DIR)


def Encoder(input_var):
	input_var = input_var.dimshuffle(0, 'x', 1, 2)

	output = lib.ops.conv2d.Conv2D(
						'Enc.1',
						input_dim=1,
						output_dim=128,
						filter_size=2,
						inputs=input_var
				)


	output = lib.ops.relu.relu(output) # shape: (batch_size, 128, 29, 29)

	output = lib.ops.conv2d.Conv2D(
						'Enc.2',
						input_dim=128,
						output_dim=128,
						filter_size=3,
						stride = 2,
						inputs=output
				)

	output = lib.ops.relu.relu(output) # shape: (batch_size, 128, 15, 15)

	output = lib.ops.conv2d.Conv2D(
						'Enc.3',
						input_dim=128,
						output_dim=128,
						filter_size=3,
						inputs=output
				)

	output = lib.ops.relu.relu(output) # shape: (batch_size, 128, 15, 15)

	output = lib.ops.conv2d.Conv2D(
						'Enc.4',
						input_dim=128,
						output_dim=128,
						filter_size=3,
						stride = 2,
						inputs=output
				)

	output = lib.ops.relu.relu(output) # shape: (batch_size, 128, 8, 8)

	output = lib.ops.conv2d.Conv2D(
						'Enc.5',
						input_dim = 128,
						output_dim = 128,
						filter_size = 3,
						mode= 'valid',
						inputs=output
				)

	output = lib.ops.relu.relu(output) # shape: (batch_size, 128, 6, 6)

	output = lib.ops.conv2d.Conv2D(
						'Enc.6',
						input_dim = 128,
						output_dim = 128,
						filter_size = 3,
						mode= 'valid',
						inputs=output
				)

	output = lib.ops.relu.relu(output) # shape: (batch_size, 128, 4, 4)


	output = lib.ops.conv2d.Conv2D(
						'Enc.7',
						input_dim = 128,
						output_dim = 128,
						filter_size = 4,
						mode = 'valid',
						inputs=output
				)

	output = lib.ops.relu.relu(output) # shape: (batch_size, 128, 1, 1)


	output = lib.ops.conv2d.Conv2D(
						'Enc.8',
						input_dim = 128,
						output_dim = 128,
						filter_size = 1,
						inputs=output
				)

	output = lib.ops.relu.relu(output) # shape: (batch_size, 128, 1, 1)

	output = lib.ops.conv2d.Conv2D(
						'Enc.9',
						input_dim = 128,
						output_dim = 128,
						filter_size = 1,
						inputs=output
				)

	output = output.reshape((output.shape[0], -1)) # shape: (batch_size, 128)
	return output[:, ::2], output[:, 1::2]

#####################################
#***** Encoder Test **************###
# inp = T.tensor3("inp")
# out = Encoder(inp)

# get_out = theano.function([inp], out)

# print(get_out(floatX(np.ones((10,28,28)))).shape)
# exit()

#####################################




def Decoder(latent_var):
	output = lib.ops.linear.Linear(
						'Dec.01',
						input_dim=64,
						output_dim=9*64,
						inputs=latent_var
				)

	output = lib.ops.relu.relu(output) # shape: (batch_size, 16*64)

	output = output.reshape((output.shape[0], 64, 3, 3)) # shape: (batch_size, 64, 3, 3)

	output = lib.ops.conv2d.Conv2D(
						'Dec.02',
						input_dim=64,
						output_dim=128,
						filter_size=3,
						inputs=output
				)

	output = lib.ops.relu.relu(output) # shape: (batch_size, 128, 3, 3)

	output = lib.ops.conv2d.Conv2D(
						'Dec.03',
						input_dim=128,
						output_dim=128,
						filter_size=3,
						inputs=output
				)

	output = lib.ops.relu.relu(output) # shape: (batch_size, 128, 3, 3)

	output = lib.ops.deconv2d.Deconv2D(
						'Dec.04',
						input_dim=128,
						output_dim=64,
						filter_size=3,
						inputs=output
				)

	output = lib.ops.relu.relu(output) # shape: (batch_size, 128, 6, 6)

	output = lib.ops.conv2d.Conv2D(
						'Dec.05',
						input_dim=64,
						output_dim=128,
						filter_size=2,
						inputs=output
				)

	output = lib.ops.relu.relu(output) # shape: (batch_size, 128, 7, 7)

	output = lib.ops.deconv2d.Deconv2D(
						'Dec.06',
						input_dim=128,
						output_dim=64,
						filter_size=3,
						inputs=output
				)

	output = lib.ops.relu.relu(output) # shape: (batch_size, 64, 14, 14)

	output = lib.ops.conv2d.Conv2D(
						'Dec.07',
						input_dim=64,
						output_dim=128,
						filter_size=3,
						inputs=output
				)

	output = lib.ops.relu.relu(output) # shape: (batch_size, 128, 14, 14)

	output = lib.ops.deconv2d.Deconv2D(
					'Dec.08',
					input_dim=128,
					output_dim=64,
					filter_size=3,
					inputs=output
			)

	output = lib.ops.relu.relu(output) # shape: (batch_size, 64, 28, 28)

	output = lib.ops.conv2d.Conv2D(
						'Dec.09',
						input_dim=64,
						output_dim=128,
						filter_size=3,
						inputs=output
				)

	output = lib.ops.relu.relu(output) # shape: (batch_size, 128, 28, 28)

	output = lib.ops.conv2d.Conv2D(
						'Dec.10',
						input_dim=128,
						output_dim=32,
						filter_size=1,
						inputs=output
				)

	output = lib.ops.relu.relu(output) # shape: (batch_size, 32, 28, 28)

	output = lib.ops.conv2d.Conv2D(
						'Dec.11',
						input_dim=32,
						output_dim=4,
						filter_size=1,
						inputs=output
				)

	output = lib.ops.relu.relu(output) # shape: (batch_size, 4, 28, 28)

	output = lib.ops.conv2d.Conv2D(
						'Dec.12',
						input_dim=4,
						output_dim=1,
						filter_size=1,
						inputs=output
				)

	output = (T.nnet.sigmoid(output) + EPS)/(floatX(1.) + EPS) # shape: (batch_size, 1, 28, 28)

	output = output.reshape((output.shape[0], 28, 28))

	return output

#####################################
# ***** Decoder Test **************###
# lv = T.matrix("lv")
# out = Decoder(lv)

# get_out = theano.function([lv], out)

# print(get_out(floatX(np.ones((10,64)))).shape)
# exit()

#####################################



def create_encoder_decoder():

	input_var = T.tensor3('input')
	input_var_normalised = (input_var - floatX(0.5))

	mu, log_square_sigma = Encoder(input_var_normalised)

	mu = lib.floatX(2.)*T.tanh(mu/lib.floatX(2.))

	sampled_z = gaussian_sampler(mu, log_square_sigma)

	reconstructed = Decoder(sampled_z)

	reconstruction_cost = T.nnet.binary_crossentropy(
								reconstructed.reshape((reconstructed.shape[0], -1)),
								input_var.reshape((input_var.shape[0], -1))
							).sum(axis=1)

	kl_cost = KL_with_standard_gaussian(mu, log_square_sigma)

	loss = T.mean(kl_cost + reconstruction_cost)

	params = lib.search(loss, lambda x: hasattr(x, 'param') and x.param==True)
	lib.print_params_info(params)

	grads = T.grad(loss, wrt=params, disconnected_inputs='warn')
	grads = [T.clip(g, lib.floatX(-1.), lib.floatX(1.)) for g in grads]

	lr = T.scalar('lr')

	updates = lasagne.updates.adam(grads, params, learning_rate=lr, epsilon=EPS)

	generated_z = T.matrix('generated_z')

	generated_samples = Decoder(generated_z)

	print "Compiling functions ..."

	train_fn = theano.function(
					[input_var, lr], 
					[loss, kl_cost.mean(), mu.min(), mu.max(), mu, sampled_z.min(), sampled_z.max()],
					updates=updates,
					# mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
				)

	reconstruct_fn = theano.function([input_var], reconstructed)

	val_fn = theano.function(
			[input_var], 
			[loss, kl_cost.mean(), mu.min(), mu.max(), mu, sampled_z.min(), sampled_z.max()],
			# mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
		)

	generate_fn = theano.function([generated_z], generated_samples)

	encode_fn = theano.function([input_var], mu)

	return train_fn, val_fn, generate_fn, reconstruct_fn, encode_fn


def run_epoch(fn, data_dict, batch_size = BATCH_SIZE, mode = 'train'):
	X = data_dict['X']
	shuffle(X)
	if 'lr' in data_dict:
		lr = data_dict['lr']
		args = [lr]
	else:
		args = []

	num_batches = len(X)//batch_size
	loss = 0.0
	kl = 0.0
	for i in range(num_batches):
		loss_, kl_, mu_min, mu_max, mu, sampled_z_min, sampled_z_max = fn(X[i::num_batches], *args)
		loss += loss_
		kl += kl_
		print "loss: {}, KL: {}, mu_min: {}, mu_max: {}, sampled_z.min : {}, sampled_z.max : {}".format(
														loss_, kl_, 
														mu_min, mu_max,
														sampled_z_min, sampled_z_max
													)
		print "Number of nans in mu: {}".format(np.count_nonzero(np.isnan(mu)))

		if (loss_ > 1e4) or np.isnan(loss_) :
			print "loss {} not allowed".format(loss_)
			exit()

	return loss/num_batches, kl/num_batches



(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train, X_test = floatX(binarize(X_train/255.0)), floatX(binarize(X_test/255.0))

lr_normal = floatX(0.001)

train_fn_normal, valid_fn_normal, generate_fn_normal, reconstruct_normal, encode_fn_normal = create_encoder_decoder()

loss_normal_train = np.zeros((100,))
loss_normal_test = np.zeros((100,))
kl_normal = np.zeros((100,))
code_variance_normal = np.zeros((100,))

print("Started Training!")

for i in range(100):
	if ((i+1)% 20) == 0:
		lr_normal = floatX(lr_normal*0.2)

	train_loss, train_kl_normal = run_epoch(train_fn_normal, {'X': X_train, 'lr': lr_normal}, mode = 'train')
	print("Epoch {}, train loss normal: {}, train KL normal: {}".format(i, train_loss, train_kl_normal))

	valid_loss, valid_kl_normal = run_epoch(valid_fn_normal, {'X': X_test}, mode = 'test')
	print("Epoch {}, valid loss normal: {}, valid KL normal: {}".format(i, valid_loss, valid_kl_normal))

	loss_normal_train[i] = train_loss

	latent_var_eps = floatX(np.random.normal(loc=0., scale= 1., size=(100, 64)))

	samples_normal = generate_fn_normal(latent_var_eps)

	test_reconstruction_normal = reconstruct_normal(X_test[0:100])
	plot_100_figure(test_reconstruction_normal, os.path.join(OUT_DIR, "reconstructed_images_normal_epoch_{}.jpg".format(i)))

	plot_100_figure(samples_normal, os.path.join(OUT_DIR, "generated_images_normal_epoch_{}.jpg".format(i)))

	codes_normal = encode_fn_normal(X_test[0:1000])
	codes_mean_normal = np.mean(codes_normal, axis = 0)
	code_variance_normal_ = np.sum((codes_normal - codes_mean_normal[None,:])**2, axis = 0)/999.

	code_variance_normal[i] = np.sum(code_variance_normal_)

	model_codes_samples_normal = floatX(np.random.multivariate_normal(codes_mean_normal, np.diag(code_variance_normal_), (100,)))
	samples_code_dist_normal = generate_fn_normal(model_codes_samples_normal)
	plot_100_figure(samples_code_dist_normal, os.path.join(OUT_DIR, "model_code_dist_normal_samples_{}.jpg".format(i)))

	plt.plot(np.arange(i+1), code_variance_normal[:i+1], 'g.-', label = "Code variance")
	plt.legend()
	plt.savefig(os.path.join(OUT_DIR, "code_variance.png"))
	plt.clf()

	plt.plot(np.arange(i+1), loss_normal_train[:i+1], 'r.-', label="Train normal")
	plt.plot(np.arange(i+1), loss_normal_test[:i+1], 'g.-', label="Test normal")

	plt.legend()

	plt.savefig(os.path.join(OUT_DIR, "loss_normal.png"))
	plt.clf()
	plt.close()






