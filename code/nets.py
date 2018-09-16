# -*- coding: utf-8 -*- 
import tensorflow as tf

class D_mlp(object):
	def __init__(self):
		self.name = "Discriminator"

	def __call__(self, input, reuse = False):
		with tf.variable_scope(self.name):
			
			D_l0 = tf.layers.dense(input, 256, tf.nn.leaky_relu, name='l0', reuse=reuse)
			D_l1 = tf.layers.dense(D_l0, 128, tf.nn.leaky_relu, name='l1', reuse=reuse)
			prob = tf.layers.dense(D_l1, 1, tf.nn.sigmoid, name='out', reuse=reuse)           
			
			# D_l3 = tf.layers.dense(G_out, 256, tf.nn.leaky_relu, name='l0', reuse=reuse)
			# D_l4 = tf.layers.dense(D_l3, 128, tf.nn.leaky_relu, name='l1', reuse=reuse)          
			# prob_artist1 = tf.layers.dense(D_l4, 1, tf.nn.sigmoid, name='out', reuse=reuse)
		return prob

class G_mlp(object):
	def __init__(self):
		self.name = "Generator"

	def __call__(self, G_in, Generator_output, reuse = True):
		with tf.variable_scope(self.name):      
			G_l1 = tf.layers.dense(G_in, 256, tf.nn.leaky_relu)
			G_l2 = tf.layers.dense(G_l1, 128, tf.nn.leaky_relu)
			G_out = tf.layers.dense(G_l2, Generator_output, tf.nn.sigmoid)
		return G_out   

#################### wGAN #######################
class D_mlp_w(object):
	def __init__(self):
		self.name = "Discriminator"

	def __call__(self, input, reuse = False):
		with tf.variable_scope(self.name):
			
			D_l0 = tf.layers.dense(input, 256, tf.nn.leaky_relu, name='l0', reuse=reuse, kernel_initializer=tf.random_normal_initializer(0,0.02))
			D_l1 = tf.layers.dense(D_l0, 128, tf.nn.leaky_relu, name='l1', reuse=reuse, kernel_initializer=tf.random_normal_initializer(0,0.02))
			prob = tf.layers.dense(D_l1, 1, activation=None, name='out', reuse=reuse, kernel_initializer=tf.random_normal_initializer(0,0.02))           
			
			# D_l3 = tf.layers.dense(G_out, 256, tf.nn.leaky_relu, name='l0', reuse=reuse)
			# D_l4 = tf.layers.dense(D_l3, 128, tf.nn.leaky_relu, name='l1', reuse=reuse)          
			# prob_artist1 = tf.layers.dense(D_l4, 1, tf.nn.sigmoid, name='out', reuse=reuse)
		return prob

class G_mlp_w(object):
	def __init__(self):
		self.name = "Generator"

	def __call__(self, G_in, Generator_output, reuse = True):
		with tf.variable_scope(self.name):      
			G_l1 = tf.layers.dense(G_in, 256, tf.nn.leaky_relu, kernel_initializer=tf.random_normal_initializer(0,0.02))
			G_l2 = tf.layers.dense(G_l1, 128, tf.nn.leaky_relu, kernel_initializer=tf.random_normal_initializer(0,0.02))
			G_out = tf.layers.dense(G_l2, Generator_output, tf.nn.sigmoid, kernel_initializer=tf.random_normal_initializer(0,0.02))
		return G_out   

#################### info_GAN #######################

class D_mlp_info(object):
	def __init__(self):
		self.name = "Discriminator"

	def __call__(self, input, Generator_output, reuse = False):
		with tf.variable_scope(self.name):
			
			D_l0 = tf.layers.dense(input, 256, tf.nn.leaky_relu, name='l0', reuse=reuse, kernel_initializer=tf.random_normal_initializer(0,0.02))
			D_l1 = tf.layers.dense(D_l0, 128, tf.nn.leaky_relu, name='l1', reuse=reuse, kernel_initializer=tf.random_normal_initializer(0,0.02))
			prob = tf.layers.dense(D_l1, 1, activation=tf.nn.sigmoid, name='out', reuse=reuse, kernel_initializer=tf.random_normal_initializer(0,0.02))           
			
			q = tf.layers.dense(D_l1, Generator_output, activation=None, name='out1', reuse=reuse, kernel_initializer=tf.random_normal_initializer(0,0.02))
			# D_l3 = tf.layers.dense(G_out, 256, tf.nn.leaky_relu, name='l0', reuse=reuse)
			# D_l4 = tf.layers.dense(D_l3, 128, tf.nn.leaky_relu, name='l1', reuse=reuse)          
			# prob_artist1 = tf.layers.dense(D_l4, 1, tf.nn.sigmoid, name='out', reuse=reuse)
		return prob, q