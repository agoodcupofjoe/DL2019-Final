#!/usr/bin/env python
# coding: utf-8

import json
import os
import glob
import sys
import shutil
from random import shuffle
from PIL import Image

def get_meta(path):
    '''
    get_meta gets the relevant clinical metadata from the txt file
    :param path: path to json file with image metadada
    :return: list of meta data types (approx_age, anatom_site_general, benign_malignant, diagnosis,
             diagnosis_confirm_type, melanocytic, sex)
    '''
    with open(path) as f:
        j = json.load(f)['meta']['clinical']
        # return list(j.values())
        return j

# get file names for malignant, benign, and others
to_remove = []
malignant = []
benign = []
for name in glob.glob("Data/Descriptions/ISIC*"):
    try:
        current_benign_malignant = get_meta(name)["benign_malignant"]
        if (current_benign_malignant == "malignant"):
            malignant.append(name[13:])
        elif (current_benign_malignant == "benign"):
            benign.append(name[13:])
        else:
            to_remove.append(name[13:])
    except:
        to_remove.append(name[13:])

# remove files from the directory if not malignant or benign
descriptions_location = "Data/Descriptions/"
images_location = "Data/Images/"
for name in to_remove:
    try:
        os.remove(descriptions_location + name)
        image_jpeg = name + ".jpeg"
        os.remove(images_location + image_jpeg)
        print("removed description {}, image {}".format(name, image_jpeg))
    except:
        print("{} might have been removed already".format(name))

# shuffle and separate data
shuffle(malignant)
shuffle(benign)
train_ratio = 0.8
num_malignant = len(malignant)
num_benign = len(benign)
train_malignant = malignant[:(int)(train_ratio * num_malignant)]
print(len(train_malignant))
test_malignant = malignant[(int)(train_ratio * num_malignant):]
print(len(test_malignant))
train_benign = benign[:(int)(train_ratio * num_benign)]
print(len(train_benign))
test_benign = benign[(int)(train_ratio * num_benign):]
print(len(test_benign))

exit()

# designate train_data and test_data
train_files = train_malignant + train_benign
print(len(train_files))
try:
    os.makedirs("Data/train/meta")
    os.makedirs("Data/train/img")
except:
    print("Train directories already exist")
for file in train_files:
    try:
        shutil.move(descriptions_location + file, "Data/train/meta/" + file)
        shutil.move(images_location + file + ".jpeg", "Data/train/img/" + file + ".jpeg")
    except:
        print("Error moving train files")
print("MOVED TRAIN DATA SUCCESSFULLY")

test_files = test_malignant + test_benign
try:
    os.makedirs("Data/test/meta")
    os.makedirs("Data/test/img")
except:
    print("Test directories already exist")
for file in test_files:
    try:
        shutil.move(descriptions_location + file, "Data/test/meta/" + file)
        shutil.move(images_location + file + ".jpeg", "Data/test/img/" + file + ".jpeg")
    except:
        print("Error moving test files")
print("MOVED TEST DATA SUCCESSFULLY")

# remove files that don't have both meta and image
# train
train_meta = glob.glob("Data/train/meta/ISIC*")
train_meta = [x[11:23] for x in train_meta]
print(len(train_meta))
train_img = glob.glob("Data/train/img/ISIC*")
train_img = [x[10:22] for x in train_img]
print(len(train_img))
train_both = list(set(train_meta) & set(train_img))
print(len(train_both))
for name in train_meta:
    if name not in train_both:
        try:
            os.remove("Data/train/meta/" + name)
        except:
            print("Error removing {}".format(name))
for name in train_img:
    if name not in train_both:
        try:
            os.remove("Data/train/img/" + name + "jpeg")
        except:
            print("Error removing {}".format(name))

# test
test_meta = glob.glob("Data/test/meta/ISIC*")
test_meta = [x[10:22] for x in test_meta]
print(len(test_meta))
test_img = glob.glob("Data/test/img/ISIC*")
test_img = [x[9:21] for x in test_img]
print(len(test_img))
test_both = list(set(test_meta) & set(test_img))
print(len(test_both))
for name in test_meta:
    if name not in test_both:
        try:
            os.remove("Data/test/meta/" + name)
        except:
            print("Error removing {}".format(name))
for name in test_img:
    if name not in test_both:
        try:
            os.remove("Data/test/img/" + name + "jpeg")
        except:
            print("Error removing {}".format(name))
