import tensorflow as tf 
import numpy as np 

def train(model, train_inputs, train_labels):



	# random shuffling of data
	rand_indx = tf.random.shuffle(np.arange(len(model.batch_size)))
	train_inputs, train_labels = tf.gather(train_inputs, rand_indx), tf.gather(train_labels, rand_indx)
	# dividing data into 5 blocks, doing different transform on each (one get no transform)
	block_size = model.batch_size // 5
	datagen = tf.keras.preprocessing.image.ImageDataGenerator().apply_transform
	# left right flip
	train_inputs[0:block_size,:,:,:] = [datagen(x, {'flip_horizontal':True}) for x in train_inputs[0:block_size,:,:,:]]
	# up down flip
	train_inputs[block_size:2*block_size,:,:,:] = [datagen(x, {'flip_vertical':True}) for x in train_inputs[block_size:2*block_size,:,:,:]]
	# random rotation
	train_inputs[2*block_size:3*block_size,:,:,:] = [datagen(x, {'theta':float(np.random.randint(360))}) for x in train_inputs[2*block_size:4*block_size,:,:,:]]
	# random shear between 0 and 20 degrees
	train_inputs[3*block_size:4*block_size,:,:,:] = [datagen(x, {'shear':float(np.random.randint(15))}) for x in train_inputs[3*block_size:4*block_size,:,:,:]]

	for img, label in zip(train_inputs, train_labels):
		with tf.GradientTape() as tape:
			logits = model.call(img)
			loss = model.loss(logits, label)
		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_inputs, test_labels):
	pass

def main():
	print('hello')

if __name__ == '__main__':
	main()
