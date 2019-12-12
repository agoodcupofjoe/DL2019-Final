from models import CNN, SENet, ResNet, SE_ResNet, ResNeXt, SE_ResNeXt
from losses import cross_entropy_loss,F1_loss, mean_F1_loss, balanced_focal_loss
from get_labels import get_one_hots_diagnosis
from cv2 import cv2

import tensorflow as tf
import numpy as np
import glob
import argparse
import os
import io
import shutil

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpu_available = tf.test.is_gpu_available()
print("GPU Available: ", gpu_available)

# argparser
parser = argparse.ArgumentParser(description='</cancer>')

parser.add_argument('--restore-checkpoint', action='store_true',
                    help='Use this flag if you want to resuming training from a previously-saved checkpoint')

parser.add_argument('--mode', type=str, default='train',
                    help='Determines whether model is in training ("train") or inference mode ("test")')

parser.add_argument('--batch-size', type=int, default=500,
                    help='Sizes of batches fed through the network')

parser.add_argument('--num-epochs', type=int, default=15,
                    help='Number of passes through the training data to make before stopping')

parser.add_argument('--learn-rate', type=float, default=0.001,
                    help='Learning rate for Adam optimizer')

parser.add_argument('--model', type=str, default='CNN',
                    help='Can be "CNN" or "SENET" or "RESNET" or "RESNEXT" or "SERESNET" or "SERESNEXT"')

parser.add_argument('--loss', type=str, default='cross_entropy',
                    help='Can be "cross_entropy" or "F1" or "mean_F1" or "focal"')

parser.add_argument('--save-output', type=bool, default=True,
                    help="Whether to save model test results to 'test_results.npz'")

parser.add_argument('--save-every', type=int, default=1,
                    help='Save the state of the network after every [this many] training iterations')

args = parser.parse_args()

# Train
def train(model, train_inputs, train_labels, manager):
    # Random shuffling of data
    rand_indx = tf.random.shuffle(np.arange(args.batch_size))
    train_inputs, train_labels = tf.gather(train_inputs, rand_indx), tf.gather(train_labels, rand_indx)

    # Dividing data into 5 blocks, doing different transform on each (one get no transform)
    block_size = args.batch_size // 5
    datagen = tf.keras.preprocessing.image.ImageDataGenerator().apply_transform

    transformed_jpegs = []

    # Left right flip
    for index in range(0, block_size):
        transformed_jpegs.append(datagen(train_inputs[index,:,:,:], {'flip_horizontal':True}))

    # Up down flip
    for index in range(block_size, 2*block_size):
        transformed_jpegs.append(datagen(train_inputs[index,:,:,:], {'flip_vertical':True}))

    # Random rotation
    for index in range(2*block_size, 3*block_size):
        transformed_jpegs.append(datagen(np.array(train_inputs[index,:,:,:]), {'theta':float(np.random.randint(360))}))

    # Random shear between 0 and 20 degrees
    for index in range(3*block_size, 4*block_size):
        transformed_jpegs.append(datagen(np.array(train_inputs[index,:,:,:]), {'shear':float(np.random.randint(20))}))

    for index in range(4*block_size, args.batch_size):
        transformed_jpegs.append(train_inputs[index,:,:,:])

    new_train_inputs = tf.stack(transformed_jpegs)

    # Determine loss function
    if args.loss == "cross_entropy":
        loss_func = cross_entropy_loss
    elif args.loss == "F1":
        loss_func = F1_loss
    elif args.loss == "mean_F1":
        loss_func = mean_F1_loss
    elif args.loss == "focal":
        loss_func = balanced_focal_loss
    
    with tf.GradientTape() as tape:
        logits = model.call(new_train_inputs, True)
        loss = loss_func(train_labels,logits)

    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Test
# Recall function
def recall(true_pos,possible_pos,epsilon=1e-7):
    return tf.dtypes.cast(true_pos, dtype = tf.float32) / (tf.dtypes.cast(possible_pos, dtype = tf.float32) + epsilon)

# Precision function
def precision(true_pos,predicted_pos,epsilon=1e-7):
        return tf.dtypes.cast(true_pos, dtype = tf.float32) / (tf.dtypes.cast(predicted_pos, dtype = tf.float32) + epsilon)

# F1 score function
def F1(recall,precision,epsilon=1e-7):
        return 2 * recall * precision / (precision + recall + epsilon)

def test(model, test_inputs, test_labels, epsilon=1e-7):

        # Pass the test images through the model's call function
    logits = model.call(test_inputs, False)
    # Determine the accuracy using the outputted logits and predictions
    accuracy = model.accuracy(logits, test_labels)
        # Model predictions
    pred = tf.argmax(logits,axis=-1)
    test_labels = tf.argmax(test_labels, axis = 1)
        # Determine true/possible/predicted positive
    true_pos = tf.math.reduce_sum(pred * tf.dtypes.cast(test_labels, dtype = tf.int64))
    possible_pos = tf.math.reduce_sum(test_labels)
    predicted_pos = tf.math.reduce_sum(pred)

    # Return the calculated accuracy and true/possible/predicted positives for
    return accuracy, true_pos, possible_pos, predicted_pos, logits, pred, test_labels

# load images in batches
def load_images(lst):
    img = []
    for file in lst:
        image = cv2.imread(file)
        img.append(image)

    return np.array(img)

# main function
def main():
    # log buffer
    log = io.StringIO()
  
    # Get names of images and load labels
    train_data = glob.glob("processed_data/train/img/*.jpeg")
    test_data = glob.glob("processed_data/test/img/ISIC_0*.jpeg")
    train_labels = get_one_hots_diagnosis("processed_data/train/meta/*")
    test_labels = get_one_hots_diagnosis("processed_data/test/meta/ISIC_0*")

    # Construct baseline model
    if args.model == "CNN":
        model = CNN()
    elif args.model == "SENET":
        model = SENet()
    elif args.model == "RESNET":
        model = ResNet()
    elif args.model == "RESNEXT":
        model = ResNeXt()
    elif args.model == "SERESNET":
        model = SE_ResNet()
    elif args.model == "SERESNEXT":
        model = SE_ResNeXt()
    print("MODEL RUNNING: {}".format(args.model))
    log.write("MODEL RUNNING: " + args.model + "\n")

    print("LOSS FUNCTION: {}\n".format(args.loss))
    log.write("LOSS FUNCTION: {}\n".format(args.loss))
    
    # For saving/loading models
    checkpoint_dir = './checkpoints/{}'.format(args.model)
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    
    # Determine whether to restore checkpoint
    if args.restore_checkpoint or args.mode == "test":
        checkpoint.restore(manager.latest_checkpoint)
    
    # Determine number of training images
    num_train = tf.shape(train_data)[0]
    indices = tf.Variable(np.arange(0, num_train, 1))
    
    if args.mode == "train":
        loss_history = []
        epoch_losses = []
        last_epoch = args.num_epochs
        # Train the baseline model for the following number of epochs
        print("*****************TRAINING*****************")
        for epoch_index in range(args.num_epochs):
            print("*****************EPOCH: {}*****************".format(epoch_index + 1))
            # Determine the number of batches to run and train
            num_batches = num_train // args.batch_size
            rand_indices = tf.random.shuffle(indices)

            # Shuffle the inputs and the labels using the shuffled indices
            train_inputs = tf.gather(train_data, rand_indices)
            train_labels = tf.gather(train_labels, rand_indices)

            epoch_loss = 0
            num_samples = 0
            for batch_index in range(num_batches):
                batch_losses = []
                # Determine the indices of the images for the current batch
                start_index = batch_index * args.batch_size
                end_index = (batch_index + 1) * args.batch_size

                # Slice and extract the current batch's data and labels
                batch_images = train_data[start_index : end_index]
                batch_labels = train_labels[start_index : end_index]
                batch_data = tf.convert_to_tensor(load_images(batch_images), dtype = tf.float32)

                # Train the model on the current batch's data and labels
                batch_loss = train(model, batch_data, batch_labels, manager)
                batch_losses.append(batch_loss)
                epoch_loss += batch_loss.numpy() * len(batch_labels)
                num_samples += len(batch_labels)
                if batch_index % 10 == 9:
                    print("TRAIN BATCH {} LOSS: {}".format(batch_index + 1,batch_loss.numpy()))
            loss_history.append(batch_losses)
            epoch_avg_loss = epoch_loss / num_samples
            with open('log/{}/{}/losses.txt'.format(args.model,args.loss),'a+') as lossfile:
                lossfile.write('EPOCH {} LOSS: {}\n'.format(epoch_index+1,epoch_avg_loss))
            if len(epoch_losses) >= 5:
                if epoch_avg_loss > np.amax(epoch_losses[-5:]):
                    last_epoch = epoch_index + 1
                    epoch_losses.append(epoch_avg_loss)
                    break
            epoch_losses.append(epoch_avg_loss)
            #manager.save()
        print("Finished training on epoch {}".format(last_epoch))

    # Determine accuracy of baseline model on test_data
    num_test = len(test_data)
    acc = 0
    truep = 0
    posp = 0
    predp = 0
    num_batches = num_test // args.batch_size
    print("")
    logits = []
    preds = []
    labels = []
    print("*****************TESTING*****************")
    for batch_index in range(num_batches + 1):
        start_index = batch_index * args.batch_size
        end_index = (batch_index + 1) * args.batch_size

        if batch_index == num_batches:
            batch_images = test_data[start_index: ]
            batch_labels = test_labels[start_index: ]
        else:
            batch_images = test_data[start_index: end_index]
            batch_labels = test_labels[start_index: end_index]

        batch_data = tf.convert_to_tensor(load_images(batch_images), dtype = tf.float32)

        a, b, c, d, e, f, g = test(model, batch_data, batch_labels)
        acc += (a * len(batch_images))
        truep += b
        posp += c
        predp += d
        logits.append(e)
        preds.append(f)
        labels.append(g)
        print("TEST BATCH: {}".format(batch_index + 1))

    # Print out evaluation metrics for the baseline model
    acc = acc / num_test
    sens = recall(truep,posp)
    prec = precision(truep,predp)
    f1 = F1(sens,prec)
    print("Global accuracy: {:.2f}%".format(acc * 100))
    log.write("Global accuracy: " + str(round((acc * 100).numpy(), 2)) + "%\n")
    print("Sensitivity: {:.2f}%".format(sens * 100))
    log.write("Sensitivity: " + str(round((sens * 100).numpy(), 2)) + "\n")
    print("Precision: {:.2f}%".format(prec * 100))
    log.write("Precision: " + str(round((prec * 100).numpy(), 2)) + "\n")
    print("F1 score: {:.2f}".format(f1))
    log.write("F1 score: " + str(round(f1.numpy(), 2)) + "\n")
    
    log_directory = "./log/" + args.model + "/" + args.loss + "/"
    try:
        os.makedirs(log_directory)
        print("Created " + log_directory)
    except:
        print("Directory already exists: " + log_directory)

    # Save and write out predictions/labels for evaluation/visualization
    if args.save_output:
        with open(log_directory + "log.txt", "w+") as fd:
            log.seek(0)
            shutil.copyfileobj(log, fd)
        log.close()
        logits = tf.concat(logits,axis=0).numpy()
        preds = tf.concat(preds,axis=0).numpy()
        labels = tf.concat(labels,axis=0).numpy()
        print("\nSAVING RESULTS")
        np.savez(log_directory + "test_results.npz",logits=logits,pred=preds,true=labels)
        if args.mode == "train":
            losses = np.array(loss_history)
            np.savez(log_directory+"training_losses.npz",losses=losses)
        print("SAVED RESULTS")

if __name__ == '__main__':
    main()
