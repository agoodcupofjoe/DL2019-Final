from baseline import Baseline
from get_labels import get_one_hots_diagnosis
from cv2 import cv2

import tensorflow as tf
import numpy as np
import glob

def train(model, train_inputs, train_labels):
	# Random shuffling of data
	rand_indx = tf.random.shuffle(np.arange(len(model.batch_size)))
	train_inputs, train_labels = tf.gather(train_inputs, rand_indx), tf.gather(train_labels, rand_indx)

	# Dividing data into 5 blocks, doing different transform on each (one get no transform)
	block_size = model.batch_size // 5
	datagen = tf.keras.preprocessing.image.ImageDataGenerator().apply_transform

	# Left right flip
	train_inputs[0:block_size,:,:,:] = [datagen(x, {'flip_horizontal':True}) for x in train_inputs[0:block_size,:,:,:]]

	# Up down flip
	train_inputs[block_size:2*block_size,:,:,:] = [datagen(x, {'flip_vertical':True}) for x in train_inputs[block_size:2*block_size,:,:,:]]

	# Random rotation
	train_inputs[2*block_size:3*block_size,:,:,:] = [datagen(x, {'theta':float(np.random.randint(360))}) for x in train_inputs[2*block_size:4*block_size,:,:,:]]

	# Random shear between 0 and 20 degrees
	train_inputs[3*block_size:4*block_size,:,:,:] = [datagen(x, {'shear':float(np.random.randint(20))}) for x in train_inputs[3*block_size:4*block_size,:,:,:]]

	for img, label in zip(train_inputs, train_labels):
		with tf.GradientTape() as tape:
			logits = model.call(img)
			loss = model.loss(logits, label)
		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_inputs, test_labels):
	baseline_logits = model.call(test_inputs)
	baseline_accuracy = model.accuracy(baseline_logits, test_labels)
	return baseline_accuracy

def load_jpegs(file_path):
	jpegs = []
	files = glob.glob(file_path)

	for file in files:
		jpeg = cv2.imread(file)
		jpegs.append(jpeg)

	return np.array(jpegs)

def main():
	#  Change file paths accordingly
	train_data = load_jpegs("processed_data/train/img/*.jpeg")

	# Need to figure out file path for labels/edit prepreprocess because
	# prepreprocess only extractsthe images and not the labels just yet
	# train_labels = get_one_hots_diagnosis()

	train_data = tf.convert_to_tensor(train_data, dtype=tf.float32)
	for index in range(tf.shape(train_data)[0]):
		print(tf.shape(train_data[index]))


	# Call function works on training data!
	baseline_model = Baseline()
	logits = baseline_model.call(train_data, True)
	print(logits)



if __name__ == '__main__':
	main()
