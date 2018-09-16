# -*- coding: utf-8 -*- 

import tensorflow as tf
import numpy as np
import os

import matplotlib.pyplot as plt

import utils as ut

import nets as net

LR_G = 0.0001
LR_D = 0.0001
BATCH_SIZE = 1
k = 1
l = 1
B = 80

# real result of loading R-tree 
def partition(size, B):    
    borders = np.random.randint(B/2, B, size=(BATCH_SIZE, size))
    borders = borders/float(B)
    return borders

class WGAN():

	def __init__(self, generator, discriminator, data_set_file):
		self.generator = generator
		self.discriminator = discriminator
		self.data_set_file = data_set_file

		indexs, latitude, longitude = ut.load_csv(self.data_set_file, 2)

		self.borders = range(B, len(latitude), B)

		self.Generator_input = 100

		self.Generator_output = len(self.borders)

		self.G_in = tf.placeholder(tf.float32, [None, self.Generator_input])

		self.real_partition = tf.placeholder(tf.float32, [None, self.Generator_output], name='real_in')

		self.G_out = self.generator(self.G_in, self.Generator_output)

		self.D_real = self.discriminator(self.real_partition)

		self.D_fake = self.discriminator(self.G_out, True)

		# self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.G_out, labels=self.real_partition)

		self.D_loss = -tf.reduce_mean(self.D_real) + tf.reduce_mean(self.D_fake)

		self.G_loss = -tf.reduce_mean(self.D_fake)

		self.train_D = tf.train.RMSPropOptimizer(LR_D).minimize(self.D_loss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.discriminator.name))

		self.train_G = tf.train.RMSPropOptimizer(LR_G).minimize(self.G_loss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.generator.name))

		self.clip_D = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.discriminator.name)]

	def train(self, training_epoches = 5000):
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		for step in range(training_epoches):
			rp = partition(len(self.borders), B)
			G_ideas = np.random.uniform(0, 1, size=(BATCH_SIZE, self.Generator_input))
			# update D 1 time
			for i in range(l):
				sess.run(self.clip_D)
				td = sess.run(self.train_D, {self.G_in: G_ideas, self.real_partition: rp})
			# update G k times
			for i in range(k):
				tg = sess.run(self.train_G, {self.G_in: G_ideas})

			# if step % 100 == 0:
			G_paintings, pa0, Dl, td, tg = sess.run([self.G_out, self.D_real, self.G_loss, self.train_D, self.train_G],{self.G_in: G_ideas, self.real_partition: rp})
			pa = pa0.mean()
			# print pa
			if step % 500 == 0:
				result = np.mean(G_paintings, axis = 0)
				result = result * B
				result = result.astype(np.int16)
				print result, pa

if __name__ == '__main__':
	generator = net.G_mlp_w()
	discriminator = net.D_mlp_w()
	data_set_file = "/Users/apple/Documents/UniMelbourne/my papers/GAN/dataset/Z.csv"
	wgan = WGAN(generator, discriminator, data_set_file)
	wgan.train()
