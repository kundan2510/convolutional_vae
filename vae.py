"""
Convolutional variational auto-encoder for MNIST.
Author: Kundan Kumar

Usage: THEANO_FLAGS='mode=FAST_RUN,device=gpu3,floatX=float32,lib.cnmem=.95' python vae.py
"""
import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer, TransposedConv2DLayer
from lasagne.layers import BatchNormLayer, ReshapeLayer
from lasagne.layers import FlattenLayer, DenseLayer
from keras.datasets import mnist
from random import shuffle

from theano.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams(seed=234)

import theano
import theano.tensor as T

import numpy as np

import os
import scipy
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from utility import *

# PREFIXES = ['/scratch/jvb-000-aa/kundan/', '/data/lisatmp4/kumarkun/']
# for p in PREFIXES:
# 	if os.path.exists(p):
# 		OUT_DIR = os.path.join(p, "mnist_conv_vae")
# 		print("Output will be saved in folder {}".format(OUT_DIR))
# 		break

OUT_DIR = "mnist_conv_vae"

if not os.path.isdir(OUT_DIR):
	print "Creating directory {}".format(OUT_DIR)
	os.makedirs(OUT_DIR)
	print "Created {}".format(OUT_DIR)


def Encoder(input_var, use_batch_norm = False):
	input_var = input_var.dimshuffle(0, 'x', 1, 2) 
	net = InputLayer(shape=(None, 1, 28, 28), input_var = input_var)

	net = Conv2DLayer(
				net, 
				num_filters=128, 
				filter_size=(2, 2),
				nonlinearity=lasagne.nonlinearities.elu
			)
	if use_batch_norm:
		net = BatchNormLayer(net)

	net = Conv2DLayer(
				net, 
				num_filters=128, 
				filter_size=(3, 3),
				stride = (2,2),
				nonlinearity=lasagne.nonlinearities.elu
			)
	if use_batch_norm:
		net = BatchNormLayer(net)
		

	net = Conv2DLayer(
            net, 
            num_filters=128, 
            filter_size=(3, 3),
			nonlinearity=lasagne.nonlinearities.elu
			)
		

	if use_batch_norm:
		net = BatchNormLayer(net)

	net = Conv2DLayer(
				net, 
				num_filters=128, 
				filter_size=(3, 3),
				stride = (2,2),
				nonlinearity=lasagne.nonlinearities.elu
			)

	if use_batch_norm:
		net = BatchNormLayer(net)
		
	net = Conv2DLayer(
				net, 
				num_filters=128, 
				filter_size=(2, 2),
				nonlinearity=lasagne.nonlinearities.elu
			)

	if use_batch_norm:
		net = BatchNormLayer(net)


	net = Conv2DLayer(
				net, 
				num_filters=128, 
				filter_size=(1, 1),
				nonlinearity=lasagne.nonlinearities.elu
			)

	net = Conv2DLayer(
				net, 
				num_filters=128, 
				filter_size=(1, 1),
				nonlinearity=lasagne.nonlinearities.elu
			)
	net = FlattenLayer(net, outdim=2)

	net = DenseLayer(net, num_units = 128, nonlinearity=lasagne.nonlinearities.rectify)

	net = DenseLayer(net, num_units = 40, nonlinearity=None)

	return net


def Decoder(latent_var, use_batch_norm = False):
	net = InputLayer(shape=(None, 20), input_var = latent_var)

	net = DenseLayer(net, num_units = 64, nonlinearity = lasagne.nonlinearities.elu)

	net = DenseLayer(net, num_units = 64*16, nonlinearity = lasagne.nonlinearities.elu)
	
	if use_batch_norm:
		net = BatchNormLayer(net)

	net = ReshapeLayer(net, (-1, 16, 8, 8))

	net = Conv2DLayer(
				net, 
				num_filters=32,
				filter_size=(3, 3),
				nonlinearity=lasagne.nonlinearities.elu
			)
	if use_batch_norm:
		net = BatchNormLayer(net)

	net = Conv2DLayer(
				net, 
				num_filters=32,
				filter_size=(3, 3),
				nonlinearity=lasagne.nonlinearities.elu
			)
	if use_batch_norm:
		net = BatchNormLayer(net)

	net = TransposedConv2DLayer(net, 8, 4, stride=(2,2), nonlinearity = lasagne.nonlinearities.elu)
	net = Conv2DLayer(
				net, 
				num_filters=32, 
				filter_size=(3, 3),
				nonlinearity=lasagne.nonlinearities.elu
			)
	if use_batch_norm:
		net = BatchNormLayer(net)

	net = TransposedConv2DLayer(net, 8, 4, stride=(2,2), nonlinearity = lasagne.nonlinearities.elu)
	net = Conv2DLayer(
				net, 
				num_filters=32, 
				filter_size=(3, 3),
				nonlinearity=lasagne.nonlinearities.elu
			)
	net = TransposedConv2DLayer(net, 8, 4, stride=(2,2), nonlinearity = lasagne.nonlinearities.elu)
	net = Conv2DLayer(
				net, 
				num_filters=32, 
				filter_size=(5, 5),
				nonlinearity=lasagne.nonlinearities.elu
			)
	if use_batch_norm:
		net = BatchNormLayer(net)

	net = Conv2DLayer(
				net, 
				num_filters=32, 
				filter_size=(3, 3),
				nonlinearity=lasagne.nonlinearities.elu
			)

	if use_batch_norm:
		net = BatchNormLayer(net)

	net = Conv2DLayer(
				net, 
				num_filters=8, 
				filter_size=(1, 1),
				nonlinearity=lasagne.nonlinearities.elu
			)
	if use_batch_norm:
		net = BatchNormLayer(net)

	net = Conv2DLayer(
				net, 
				num_filters=1, 
				filter_size=(1, 1),
				nonlinearity=lasagne.nonlinearities.sigmoid
			)

	return net


def run_epoch(fn, data_dict, batch_size = 1000, mode = 'train'):
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
		loss_, kl_ = fn(X[i::num_batches], *args)
		loss += loss_
		kl += kl_
		print loss_, kl_

	return loss/num_batches, kl/num_batches

def create_encoder_decoder(input_var, lr=0.001, use_batch_norm = False):
	""" 
	Generates functions for training, validation and generating samples. Build entire 
	computational graph and compiles theano functions
	"""
	lr  = T.scalar('lr') # Needed if you want learning rate decay

	print "Creating computational graph"

	# normalising input to have mean 0. and std 1.
	input_var_normalised = (input_var - floatX(0.5)) #input_var assumed to be in range 0 to 1


	# creating encoder function
	encoder_net = Encoder(input_var_normalised, use_batch_norm = use_batch_norm)

	# This gives the parameters of the postirior distribution of latent variables given input
	# Since we have assumed that the postirior takes gaussian distribution, we find 
	# mean and variance of the distribution

	latent_var_stats = lasagne.layers.get_output(encoder_net)

	# At test time we use the batchnorm stats for every layer computed during 
	# training, hence we pass deterministic = True
	latent_var_stats_test = lasagne.layers.get_output(encoder_net, deterministic=True)

	mu_z, log_sigma_z = latent_var_stats[:,::2], latent_var_stats[:,1::2]

	mu_z_test, log_sigma_z_test = latent_var_stats_test[:,::2], latent_var_stats_test[:,1::2]


	# We assume a standard gaussian prior, hence compute the KL between posterior and standard gaussian
	KL = KL_with_standard_gaussian(mu_z, log_sigma_z)

	KL_test = KL_with_standard_gaussian(mu_z_test, log_sigma_z_test)


	# We sample using reparametrization trick
	sampled_z = gaussian_sampler(mu_z, log_sigma_z)

	sampled_z_test = gaussian_sampler(mu_z_test, log_sigma_z_test)

	# Following variable is required so that we can sample from prior
	z_generation = T.matrix('z_generation')

	decoder_net = Decoder(sampled_z, use_batch_norm = use_batch_norm)

	reconstructed_input = lasagne.layers.get_output(decoder_net)

	generated_output = lasagne.layers.get_output(decoder_net, inputs = z_generation, deterministic=True)
	generated_output = generated_output.reshape((generated_output.shape[0], generated_output.shape[2], generated_output.shape[3]))
	
	reshaped_reconstruction = reconstructed_input.reshape((reconstructed_input.shape[0], reconstructed_input.shape[2], reconstructed_input.shape[3]))

	# Single sample monte-carlo estimate of reconstruction cost and KL. This is variational lower bound.
	train_loss = (T.nnet.binary_crossentropy(reshaped_reconstruction, input_var)).mean(axis=0).sum() + KL.mean()

	params_encoder = lasagne.layers.get_all_params(encoder_net, trainable=True)
	params_decoder = lasagne.layers.get_all_params(decoder_net, trainable=True)

	# params = params_decoder + params_encoder + params_latent
	params = params_decoder + params_encoder

	grads = T.grad(train_loss, wrt=params, disconnected_inputs='warn')
	grads = [T.clip(g, floatX(-1.), floatX(1.)) for g in grads]

	updates = lasagne.updates.adam(grads, params, learning_rate=lr)


	test_output = lasagne.layers.get_output(decoder_net, inputs = sampled_z_test, deterministic=True)

	test_output = test_output.reshape((test_output.shape[0], test_output.shape[2], test_output.shape[3]))
	test_loss = T.nnet.binary_crossentropy(test_output, input_var).mean(axis=0).sum() + KL_test.mean()

	print "Compiling functions"
	# This will be used for training
	train_fn = theano.function([input_var, lr], [train_loss, KL.mean()], updates=updates)

	# This will be used for generating reconstructions
	reconstruct_fn = theano.function([input_var], test_output)

	# This will be used for validation
	val_fn = theano.function([input_var], [test_loss, KL_test.mean()])

	# This will be used for generating samples from the model
	generate_fn = theano.function([z_generation], generated_output)


	# This function will be used for getting mean of posterior distribution of latent variables
	encode_fn = theano.function([input_var], mu_z_test)

	print "All functions compiled."

	return train_fn, val_fn, generate_fn, reconstruct_fn, encode_fn

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train, X_test = floatX(binarize(X_train/255.0)), floatX(binarize(X_test/255.0))

lr_bn = floatX(0.00001)
lr_normal = floatX(0.00001)

input_var = T.tensor3('input')

train_fn_normal, valid_fn_normal, generate_fn_normal, reconstruct_normal, encode_fn_normal = create_encoder_decoder(input_var, lr = lr_normal)

train_fn_bn, valid_fn_bn, generate_fn_bn, reconstruct_bn, encode_fn_bn = create_encoder_decoder(input_var, lr = lr_bn, use_batch_norm = True)

loss_bn_train = np.zeros((100,))
loss_normal_train = np.zeros((100,))
loss_bn_test = np.zeros((100,))
loss_normal_test = np.zeros((100,))

kl_normal = np.zeros((100,))
kl_bn = np.zeros((100,))


mu_s_variance_bn = np.zeros((100,))
mu_s_variance_normal = np.zeros((100,))

code_variance_bn = np.zeros((100,))
code_variance_normal = np.zeros((100,))

print("Started Training!")

for i in range(100):
	if ((i+1)% 20) == 0:
		lr_bn = floatX(lr_bn*0.2)
		lr_normal = floatX(lr_normal*0.2)

	train_loss, train_kl_normal = run_epoch(train_fn_normal, {'X': X_train, 'lr': lr_normal}, mode = 'train')
	print("Epoch {}, train loss normal: {}, train KL normal: {}".format(i, train_loss, train_kl_normal))

	valid_loss, valid_kl_normal = run_epoch(valid_fn_normal, {'X': X_test}, mode = 'test')
	print("Epoch {}, valid loss normal: {}, valid KL normal: {}".format(i, valid_loss, valid_kl_normal))


	train_loss_bn, train_kl_bn = run_epoch(train_fn_bn, {'X': X_train, 'lr': lr_normal}, mode = 'train')
	print("Epoch {}, train loss BN: {}, train KL BN: {}".format(i, train_loss_bn, train_kl_bn))

	valid_loss_bn, valid_kl_bn = run_epoch(valid_fn_bn, {'X': X_test}, mode = 'test')
	print("Epoch {}, valid loss BN: {}, , train KL BN: {}".format(i, valid_loss_bn, valid_kl_bn))

	loss_normal_train[i] = train_loss
	loss_bn_train[i] = train_loss_bn

	loss_normal_test[i] = valid_loss
	loss_bn_test[i] = valid_loss_bn

	latent_var_eps = floatX(np.random.normal(loc=0., scale= 1., size=(100, 20)))

	samples_bn = generate_fn_bn(latent_var_eps)
	print samples_bn.shape

	test_reconstruction_bn = reconstruct_bn(X_test[0:100])

	plot_100_figure(test_reconstruction_bn, os.path.join(OUT_DIR, "reconstructed_images_bn_epoch_{}.jpg".format(i)))

	plot_100_figure(samples_bn, os.path.join(OUT_DIR, "generated_images_bn_epoch_{}.jpg".format(i)))


	samples_normal = generate_fn_bn(latent_var_eps)

	test_reconstruction_normal = reconstruct_normal(X_test[0:100])
	plot_100_figure(test_reconstruction_normal, os.path.join(OUT_DIR, "reconstructed_images_normal_epoch_{}.jpg".format(i)))

	plot_100_figure(samples_normal, os.path.join(OUT_DIR, "generated_images_normal_epoch_{}.jpg".format(i)))



	codes_bn = encode_fn_bn(X_test[0:1000])
	codes_mean_bn = np.mean(codes_bn, axis = 0)
	code_variance_bn_ = np.sum((codes_bn - codes_mean_bn[None,:])**2, axis = 0)/999.

	codes_normal = encode_fn_normal(X_test[0:1000])
	codes_mean_normal = np.mean(codes_normal, axis = 0)
	code_variance_normal_ = np.sum((codes_normal - codes_mean_normal[None,:])**2, axis = 0)/999.

	code_variance_normal[i] = np.sum(code_variance_normal_)
	code_variance_bn[i] = np.sum(code_variance_bn_)

	model_codes_samples_bn = floatX(np.random.multivariate_normal(codes_mean_bn, np.diag(code_variance_bn_), (100,)))
	samples_code_dist_bn = generate_fn_bn(model_codes_samples_bn)
	plot_100_figure(samples_code_dist_bn, os.path.join(OUT_DIR, "model_code_dist_bn_samples_{}.jpg".format(i)))

	model_codes_samples_normal = floatX(np.random.multivariate_normal(codes_mean_normal, np.diag(code_variance_normal_), (100,)))
	samples_code_dist_normal = generate_fn_normal(model_codes_samples_normal)
	plot_100_figure(samples_code_dist_normal, os.path.join(OUT_DIR, "model_code_dist_normal_samples_{}.jpg".format(i)))

	plt.plot(np.arange(i+1), code_variance_bn[:i+1], 'r.-', label = "BN")
	plt.plot(np.arange(i+1), code_variance_normal[:i+1], 'g.-', label = "No BN")
	plt.legend()
	plt.savefig(os.path.join(OUT_DIR, "code_variance.png"))
	plt.clf()

	plt.plot(np.arange(i+1), loss_normal_train[:i+1], 'r.-', label="Train normal")
	plt.plot(np.arange(i+1), loss_normal_test[:i+1], 'g.-', label="Test normal")

	plt.plot(np.arange(i+1), loss_bn_train[:i+1], 'r*-', label="Train BN")
	plt.plot(np.arange(i+1), loss_bn_test[:i+1], 'g*-', label="Test BN")
	plt.legend()

	plt.savefig(os.path.join(OUT_DIR, "loss_normal_bn.png"))
	plt.clf()
	plt.close()

