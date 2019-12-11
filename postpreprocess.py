import glob
import json
import sys
import shutil
from random import shuffle
from PIL import Image

'''
get_meta gets the relevant clinical metadata from the txt file
:param path: path to json file with image metadada
:return: list of meta data types (approx_age, anatom_site_general, benign_malignant, diagnosis,
         diagnosis_confirm_type, melanocytic, sex)
'''
def get_meta(path):
    with open(path) as f:
        j = json.load(f)['meta']['clinical']
        # return list(j.values())
        return j

'''
convert changes a name of ISIC file to a new file name to be copied as
:param s: string of original file name
:i: number for the i-th copy
'''
def convert(s, i):
    return str(s)[:-7] + str(i) + str(s)[-6:]

'''
get names of "malignant" files
'''
malignant_train = []
for name in glob.glob("processed_data/train/meta/ISIC*"):
     current_benign_malignant = get_meta(name)["benign_malignant"]
     if (current_benign_malignant == "malignant"):
         malignant_train.append(name[-12:])
print(len(malignant_train))

'''
copy images five times
'''
train_path = "processed_data/train/"
to_iterate = [1, 2, 3, 4, 5]
for i in to_iterate:
    for name in malignant_train:
        try:
            meta_name = train_path + "meta/" + name
            shutil.copy(meta_name, convert(meta_name, i))
            img_name = train_path + "img/" + name
            shutil.copy(img_name + ".jpeg", convert(img_name, i) + ".jpeg")
        except:
            pass
