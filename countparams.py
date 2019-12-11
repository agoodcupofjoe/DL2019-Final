from models import CNN,SENet,ResNet,SE_ResNet,ResNeXt,SE_ResNeXt
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf

def printparams(model,name,inputs):
    a = model.call(inputs,True)
    trainable = np.sum([K.count_params(w) for w in model.trainable_weights])
    print("{} trainable parameters: {:,}".format(name,trainable))
    pass

def main():
    names = ["CNN","SENet","ResNet","SE-ResNet","ResNeXt","SE-ResNeXt"]
    models = [CNN(),SENet(),ResNet(),SE_ResNet(),ResNeXt(),SE_ResNeXt()]
    inputs = tf.zeros([1,150,200,3])
    for i in range(len(names)):
        printparams(models[i],names[i],inputs)

if __name__ == "__main__":
    main()
    
