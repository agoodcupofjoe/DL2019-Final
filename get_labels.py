import json
import glob
import numpy as np
import tensorflow as tf

def get_meta(path):
    '''
    get_meta gets the relevant clinical metadata from the txt file
    :param path: path to json file with image metadada
    :return: list of meta data types (approx_age, anatom_site_general, benign_malignant, diagnosis,
             diagnosis_confirm_type, melanocytic, sex)
    '''
    with open(path) as f:
        j = json.load(f)['meta']['clinical']
        return j

def get_one_hots_diagnosis(directory_path):
    '''
    This fn gets a one-hot tensor ([0,1] if benign, [1,0] if malignant)

    params: directory_path, the path to the directory containing all of the labels, with /* added to the end

    output: labels, a one-hot tensor of the labels for whether the img is malignant or benign

    '''
    files = glob.glob(directory_path)
    files = np.sort(files)

    labels = [get_meta(str(file))['benign_malignant'] == 'benign' for file in files]
    labels = tf.one_hot(labels, 2)

    return labels