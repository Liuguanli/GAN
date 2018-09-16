import tensorflow as tf
from gan_demo import GAN
import utils as ut
import numpy as np
import nets as net

LR_G = 0.0001
LR_D = 0.0001
BATCH_SIZE = 1
k = 1
B = 80

# real result of loading R-tree 
def partition(size, B):    
    borders = np.random.randint(B/2, B, size=(BATCH_SIZE, size))
    borders = borders/float(B)
    return borders

def concat(z,y):
	return tf.concat([z,y],1)

class CGAN():
	def __init__(self, generator, discriminator, data_set_file, y_dim):
		self.generator = generator
		self.discriminator = discriminator
		self.data_set_file = data_set_file
		self.y_dim = y_dim # condition

		indexs, latitude, longitude = ut.load_csv(self.data_set_file, 2)

		self.borders = range(B, len(latitude), B)

		self.Generator_input = 100

		self.Generator_output = len(self.borders)

		self.G_in = tf.placeholder(tf.float32, [None, self.Generator_input])
		self.real_partition = tf.placeholder(tf.float32, [None, self.Generator_output], name='real_in')
		self.condition = tf.placeholder(tf.float32, shape=[None, self.y_dim])

		self.G_out = self.generator(concat(self.G_in, self.condition), self.Generator_output)

		self.D_real = self.discriminator(concat(self.real_partition, self.condition))

		self.D_fake  = self.discriminator(concat(self.G_out, self.condition), reuse = True)

		self.D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real, labels=tf.ones_like(self.D_real))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.zeros_like(self.D_fake)))

		self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.ones_like(self.D_fake)))

		self.train_D = tf.train.AdamOptimizer(LR_D).minimize(self.D_loss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.discriminator.name))

		self.train_G = tf.train.AdamOptimizer(LR_G).minimize(self.G_loss,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.generator.name))

	def train(self, training_epoches = 5000):
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		for step in range(training_epoches):
			rp = partition(len(self.borders), B)
			G_ideas = np.random.uniform(0, 1, size=(BATCH_SIZE, self.Generator_input))
			condition = np.random.uniform(0, 2, size=(BATCH_SIZE, self.y_dim))
			# update D 1 time
			td = sess.run(self.train_D, {self.G_in: G_ideas, self.real_partition: rp, self.condition: condition})
			# update G k times
			for i in range(k):
				tg = sess.run(self.train_G, {self.G_in: G_ideas, self.condition: condition})

			# if step % 100 == 0:
			G_paintings, pa0, Dl, td, tg = sess.run([self.G_out, self.D_real, self.G_loss, self.train_D, self.train_G],{self.G_in: G_ideas, self.real_partition: rp, self.condition: condition})
			pa = pa0.mean()
			# print pa
			if 0.499 < pa < 0.501:
				result = np.mean(G_paintings, axis = 0)
				result = result * B
				result = result.astype(np.int16)
				print result, pa

if __name__ == '__main__':
	generator = net.G_mlp()
	discriminator = net.D_mlp()
	data_set_file = "../dataset/Z.csv"
	cgan = CGAN(generator, discriminator, data_set_file, 20)
	cgan.train()