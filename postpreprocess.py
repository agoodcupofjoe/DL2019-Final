import glob
import json
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

def convert(s):
    return str(s)[:-7] + "1" + str(s)[-6:]

malignant_train = []
for name in glob.glob("processed_data/train/meta/ISIC*"):
     current_benign_malignant = get_meta(name)["benign_malignant"]
     if (current_benign_malignant == "malignant"):
         malignant_train.append(name[-12:])
print(len(malignant_train))

train_path = "processed_data/train/"
for name in malignant_train:
    try:
        meta_name = train_path + "meta/" + name
        shutil.copy(meta_name, convert(meta_name))
        img_name = train_path + "img/" + name
        shutil.copy(img_name + ".jpeg", convert(img_name) + ".jpeg")
    except:
        pass

malignant_test = []
for name in glob.glob("processed_data/test/meta/ISIC*"):
    current_benign_malignant = get_meta(name)["benign_malignant"]
    if (current_benign_malignant == "malignant"):
        malignant_test.append(name[-12:])
print(len(malignant_test))

test_path = "processed_data/test/"
for name in malignant_test:
    try:
        meta_name = test_path + "meta/" + name
        shutil.copy(meta_name, convert(meta_name))
        img_name = test_path + "img/" + name
        shutil.copy(img_name + ".jpeg", convert(img_name) + ".jpeg")
    except:
        pass
