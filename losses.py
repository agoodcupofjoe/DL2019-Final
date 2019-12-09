import tensorflow as tf

def F1_loss(labels,pred,from_logits=True,epsilon=1e-7):
    labels = tf.cast(labels,tf.float32)
    if from_logits:
        pred = tf.nn.softmax(pred,axis=-1)
    truepos = tf.reduce_sum(labels * pred,axis=0)
    falsepos = tf.reduce_sum((1-labels) * pred,axis=0)
    falseneg = tf.reduce_sum(labels * (1-pred),axis=0)
    F1 = 2 * truepos / (2 * truepos + falsepos + falseneg + epsilon)
    F1 = tf.where(tf.math.is_nan(F1),tf.zeros_like(F1),F1)
    return 1 - tf.reduce_mean(F1)

def mean_F1_loss(labels,pred,from_logits=True,epsilon=1e-7):
    labels = tf.cast(labels,tf.float32)
    if from_logits:
        pred = tf.nn.softmax(pred,axis=-1)
    truepos = tf.reduce_sum(labels * pred,axis=0)
    trueneg = tf.reduce_sum((1-labels) * (1-pred),axis=0)
    falsepos = tf.reduce_sum((1-labels) * pred,axis=0)
    falseneg = tf.reduce_sum(labels * (1-pred),axis=0)
    F1_pos = 2 * truepos / (2 * truepos + falsepos + falseneg + epsilon)
    F1_neg = 2 * trueneg / (2 * trueneg + falseneg + falsepos + epsilon)
    F1_pos = tf.where(tf.math.is_nan(F1_pos),tf.zeros_like(F1_pos),F1_pos)
    F1_neg = tf.where(tf.math.is_nan(F1_neg),tf.zeros_like(F1_neg),F1_neg)
    F1 = (F1_pos + F1_neg) / 2.0
    return 1 - tf.reduce_mean(F1)

def balanced_focal_loss(labels,pred,lam=2,alpha=0.25,from_logits=True,epsilon=1e-7):
    '''Reference: "Focal Loss for Dense Object Detection" by Lin, Goyal, Girshick, He, and Dollar'''
    labels = tf.cast(labels,tf.float32)
    if from_logits:
        pred = tf.nn.softmax(pred,axis=-1)
    pc = pred * labels
    focal = -alpha * tf.pow(1 - pc,lam) * tf.math.log(pc)
    focal = tf.where(tf.math.is_nan(focal),tf.zeros_like(focal),focal)
    return tf.reduce_mean(focal)
