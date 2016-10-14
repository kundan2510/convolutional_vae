import lasagne
from lasagne.layers import InputLayer, FlattenLayer, DenseLayer
from lasagne.layers import Conv2DLayer, TransposedConv2DLayer
from lasagne.layers import BatchNormLayer, ReshapeLayer
from lasagne.layers import Pool2DLayer
from keras.datasets import mnist

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

OUT_DIR = "/data/lisatmp4/kumarkun/mnist_conv_vae"

if not os.path.isdir(OUT_DIR):
	os.makedirs(OUT_DIR)

def floatX(a):
	return np.float32(a)

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


def Encoder(input_var, use_batch_norm = False):
	input_var = input_var.dimshuffle(0, 'x', 1, 2) 
	net = InputLayer(shape=(None, 1, 28, 28), input_var = input_var)

	net = Conv2DLayer(
				net, 
				num_filters=32, 
				filter_size=(2, 2),
				nonlinearity=lasagne.nonlinearities.elu,
				W=lasagne.init.GlorotUniform()
			)
	if use_batch_norm:
		net = BatchNormLayer(net)

	net = Conv2DLayer(
				net, 
				num_filters=64, 
				filter_size=(3, 3),
				stride = (2,2),
				nonlinearity=lasagne.nonlinearities.elu,
				W=lasagne.init.GlorotUniform()
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
				nonlinearity=lasagne.nonlinearities.elu,
				W=lasagne.init.GlorotUniform()
			)

	if use_batch_norm:
		net = BatchNormLayer(net)
		
	net = Conv2DLayer(
				net, 
				num_filters=128, 
				filter_size=(2, 2),
				nonlinearity=lasagne.nonlinearities.elu,
				W=lasagne.init.GlorotUniform()
			)

	if use_batch_norm:
		net = BatchNormLayer(net)


	net = Conv2DLayer(
				net, 
				num_filters=2, 
				filter_size=(1, 1),
				nonlinearity=None,
				W=lasagne.init.GlorotUniform()
			)

	return net


def Decoder(latent_var, use_batch_norm = False):
	net = InputLayer(shape=(None,1, 4, 4), input_var = latent_var)

	net = TransposedConv2DLayer(net, 16, 4, stride=(2,2), nonlinearity = lasagne.nonlinearities.elu)

	net = Conv2DLayer(
				net, 
				num_filters=32,
				filter_size=(3, 3),
				nonlinearity=lasagne.nonlinearities.elu
			)

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


def run_epoch(fn, data_dict, batch_size = 100, mode = 'train'):
	X = data_dict['X']
	if 'lr' in data_dict:
		lr = data_dict['lr']
		args = [lr]
	else:
		args = []

	num_batches = len(X)//batch_size
	loss = 0.0
	kl = 0.0
	for i in range(num_batches):
		loss_, kl_ = fn(X[i*batch_size:(i+1)*batch_size, :, :], *args)
		loss += loss_
		kl += kl_

	return loss/num_batches, kl/num_batches
	
def KL_with_standard_gaussian(mu, var):
	return (
		np.float32(-0.5) * (np.float32(1.) +
		np.float32(2.) * log_sigma - mu**np.float32(2.) - 
		T.exp(np.float32(2.) * log_sigma)
	).mean(axis=0).sum()


def gaussian_sampler(mu, log_sigma):
	eps = T.cast(srng.normal(size=mu.shape, std=1.), theano.config.floatX)
	return mu + T.exp(log_sigma)*eps


def create_encoder_decoder(input_var, lr=0.001, use_batch_norm = False):
	lr  = T.scalar('lr')

	input_var_normalised = (input_var - floatX(0.5))*floatX(2*1.732) #input_var assumed to be in range 0 to 1

	encoder_net = Encoder(input_var_normalised, use_batch_norm = use_batch_norm)

	
	latent_var_stats = lasagne.layers.get_output(encoder_net)
	latent_var_stats_test = lasagne.layers.get_output(encoder_net, deterministic=True)

	mu_z, log_sigma_z = latent_var_stats[:,0:1], latent_var_stats[:,1:2]

	mu_z_test, log_sigma_z_test = latent_var_stats_test[:,0:1], latent_var_stats_test[:,1:2]

	KL = KL_with_standard_gaussian(mu_z, log_sigma_z)

	KL_test = KL_with_standard_gaussian(mu_z_test, log_sigma_z_test)

	sampled_z = gaussian_sampler(mu_z, log_sigma_z)

	sampled_z_test = gaussian_sampler(mu_z_test, log_sigma_z_test)

	z_generation = T.tensor3('z_generation')
	z_generation_4d = z_generation.dimshuffle(0,'x', 1, 2)

	decoder_net = Decoder(sampled_z, use_batch_norm = use_batch_norm)

	reconstructed_input = lasagne.layers.get_output(decoder_net)

	generated_output = lasagne.layers.get_output(decoder_net, inputs = z_generation_4d, deterministic=True)
	
	reshaped_reconstruction = target_var_train.reshape((target_var_train.shape[0], target_var_train.shape[2], target_var_train.shape[3]))

	train_loss = T.mean(T.nnet.binary_crossentropy(reshaped_reconstruction, input_var)) + KL

	params_encoder = lasagne.layers.get_all_params(encoder_net, trainable=True)
	params_decoder = lasagne.layers.get_all_params(decoder_net, trainable=True)

	# params = params_decoder + params_encoder + params_latent
	params = params_decoder + params_encoder

	updates = lasagne.updates.nesterov_momentum(
				train_loss, params, learning_rate = lr, momentum=0.9
				)

	test_output = lasagne.layers.get_output(decoder_net, inputs = sampled_z_test, deterministic=True)

	test_output = test_output.reshape((test_output.shape[0], test_output.shape[2], test_output.shape[3]))
	test_loss = T.mean(T.nnet.binary_crossentropy(test_output, input_var)) + KL_test

	train_fn = theano.function([input_var, lr], [train_loss, KL], updates=updates)

	reconstruct_fn = theano.function([input_var], test_output)

	val_fn = theano.function([input_var], [test_loss, KL_test])

	generate_fn = theano.function([z_generation], generated_output)

	encode_fn = theano.function([input_var], mu_z_test)

	return train_fn, val_fn, generate_fn, reconstruct_fn, encode_fn

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train, X_test = binarize(X_train/255.0).astype('float32'), binarize(X_test/255.0).astype('float32')


lr_bn = floatX(0.1)
lr_normal = floatX(0.01)

input_var = T.tensor3('input')


####### Just for testing
# enc_net = Encoder(input_var, use_batch_norm = False)
# latent_var_train = lasagne.layers.get_output(enc_net)

# shape_fn = theano.function([input_var], latent_var_train)

# print shape_fn(X_train[0:2]).shape
# exit()

#######
###########  Just for testing

# l_var = T.tensor4('l_var')
# dec_net = Decoder(l_var)
# net_out = lasagne.layers.get_output(dec_net)

# shape_fn = theano.function([l_var], net_out)
# print shape_fn(np.ones((5,1,4,4)).astype('float32')).shape
# exit()


##########




# mean = T.tensor3('mean')
# log_sigma = T.tensor3('log_sigma')

# shape_fn = theano.function([mean, log_sigma], KL_with_standard_gaussian(mean, log_sigma))

# print shape_fn(np.zeros((5,4,4)).astype('float32'), np.ones((5,4,4)).astype('float32')).shape
# exit()

##########

train_fn_normal, valid_fn_normal, generate_fn_normal, reconstruct_normal, encode_fn_normal = create_encoder_decoder(input_var)

train_fn_bn, valid_fn_bn, generate_fn_bn, reconstruct_bn, encode_fn_bn = create_encoder_decoder(input_var, use_batch_norm = True)

loss_bn_train = np.zeros((100,))
loss_normal_train = np.zeros((100,))
loss_bn_test = np.zeros((100,))
loss_normal_test = np.zeros((100,))

mu_s_variance_bn = np.zeros((100,))
mu_s_variance_normal = np.zeros((100,))

code_variance_bn = np.zeros((100,))
code_variance_normal = np.zeros((100,))

print("Started Training!")

for i in range(100):
	if ((i+1)% 20) == 0:
		lr_bn = floatX(lr_bn*0.2)
		lr_normal = floatX(lr_normal*0.2)

	train_loss = run_epoch(train_fn_normal, {'X': X_train, 'lr': lr_normal}, mode = 'train')
	print("Epoch {}, train loss normal: {}".format(i, train_loss))

	valid_loss = run_epoch(valid_fn_normal, {'X': X_test}, mode = 'test')
	print("Epoch {}, valid loss normal: {}".format(i, valid_loss))


	train_loss_bn = run_epoch(train_fn_bn, {'X': X_train, 'lr': lr_normal}, mode = 'train')
	print("Epoch {}, train loss BN: {}".format(i, train_loss_bn))

	valid_loss_bn = run_epoch(valid_fn_bn, {'X': X_test}, mode = 'test')
	print("Epoch {}, valid loss BN: {}".format(i, valid_loss_bn))

	loss_normal_train[i] = train_loss
	loss_bn_train[i] = train_loss_bn

	loss_normal_test[i] = valid_loss
	loss_bn_test[i] = valid_loss_bn

	latent_var_eps = np.random.normal(loc=0., scale= 1., size=(100, 1, 4, 4)).astype('float32')

	samples_bn = generate_fn_bn(latent_var_eps)

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

	model_codes_samples_bn = np.random.multivariate_normal(codes_mean_bn, np.diag(code_variance_bn_), (100,)).astype('float32')
	samples_code_dist_bn = generate_fn_bn(model_codes_samples_bn)
	plot_100_figure(samples_code_dist_bn, os.path.join(OUT_DIR, "model_code_dist_bn_samples_{}.jpg".format(i)))

	model_codes_samples_normal = np.random.multivariate_normal(codes_mean_normal, np.diag(code_variance_normal_), (100,)).astype('float32')
	samples_code_dist_normal = generate_fn_normal(model_codes_samples_normal)
	plot_100_figure(samples_code_dist_normal, os.path.join(OUT_DIR, "model_code_dist_normal_samples_{}.jpg".format(i)))

	plt.plot(np.arange(i+1), code_variance_bn[:i+1], 'r.-', label = "BN")
	plt.plot(np.arange(i+1), code_variance_normal[:i+1], 'g.-', label = "No BN")
	plt.legend()
	plt.savefig(os.path.join(OUT_DIR, "code_variance.png"))
	plt.clf()

	plt.plot(np.arange(i+1), loss_normal_train[:i+1], 'r.-')
	plt.plot(np.arange(i+1), loss_normal_test[:i+1], 'g.-')

	plt.plot(np.arange(i+1), loss_bn_train[:i+1], 'r*-')
	plt.plot(np.arange(i+1), loss_bn_test[:i+1], 'g*-')

	plt.savefig(os.path.join(OUT_DIR, "loss_normal_bn.png"))
	plt.clf()

