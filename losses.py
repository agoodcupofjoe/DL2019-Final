import tensorflow as tf

def cross_entropy_loss(labels,pred,from_logits=True):
    if from_logits:
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels,pred))
    else:
        loss = tf.keras.losses.binary_crossentropy(labels,pred)
    return loss

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

def balanced_focal_loss(labels,pred,gamma=2,alpha=0.25,from_logits=True,epsilon=1e-7):
    '''Reference: "Focal Loss for Dense Object Detection" by Lin, Goyal, Girshick, He, and Dollar'''
    labels = tf.cast(labels,tf.float32)
    if from_logits:
        pred = tf.nn.softmax(pred,axis=-1)
    p0 = tf.where(tf.equal(labels,0),pred,tf.zeros_like(pred))
    p1 = tf.where(tf.equal(labels,1),pred,tf.ones_like(pred))
    p0 = tf.clip_by_value(p0,epsilon,1. - epsilon)
    p1 = tf.clip_by_value(p1,epsilon,1. - epsilon)
    focal1 = alpha * tf.pow(1. - p1,gamma) * tf.math.log(p1)
    focal0 = (1 - alpha) * tf.pow(1. - p0,gamma) * tf.math.log(1. - p0)
    return -tf.reduce_mean(focal1 + focal0)
