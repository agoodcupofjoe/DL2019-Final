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
	# Pass the test images through the model's call function
	logits = model.call(test_inputs)

	# Determine the accuracy using the outputted logits and predictions
	accuracy = model.accuracy(logits, test_labels)

	# Return the calculated accuracy
	return accuracy

def load_jpegs(file_path):
	jpegs = []
	files = glob.glob(file_path)

	for file in files:
		jpeg = cv2.imread(file)
		jpegs.append(jpeg)

	return np.array(jpegs)

def main():
	# Acquire and load training images as tensors
	train_data = load_jpegs("processed_data/train/img/*.jpeg")
	train_data = tf.convert_to_tensor(train_data, dtype=tf.float32)

	# Acquire the labels for the corresponding training images
	train_labels = get_one_hots_diagnosis("processed_data/train/labels/*")

	# Acquire and load testing images as tensors
	test_data = load_jpegs("processed_data/test/img/*.jpeg")
	test_data = tf.convert_to_tensor(test_data, dtype=tf.float32)

	# Acquire the labels for the corresponding testing images
	test_labels = get_one_hots_diagnosis("process_data/test/labels/*")

	# Construct baseline model
	baseline_model = Baseline()

	# Determine number of training images
	num_train = tf.shape(train_data)[0]

	# Train the baseline model for the following number of epochs
	for epoch_index in range(baseline_model.num_epochs):
		# Determine the number of batches to run and train
		num_batches = (num_train // baseline_model.batch_size)

		for batch_index in range(num_batches):
			# Determine the indices of the images for the current batch
			start_index = batch_index * model.batch_size
			end_index = (batch_index + 1) * model.batch_size

			# Slice and extract the current batch's data and labels
			batch_data = train_data[start_index:end_index]
			batch_labels = train_labels[start_index:end_index]

			# Train the model on the current batch's data and labels
			train(baseline_model, batch_data, batch_labels)

	# Determine accuracy of baseline model on test_data
	baseline_accuracy = test(baseline_model, test_data, test_labels) * 100

	# Print out the accuracy of the baseline model
	print(f"Baseline Model Accuracy: {baseline_accuracy}%")



if __name__ == '__main__':
	main()
