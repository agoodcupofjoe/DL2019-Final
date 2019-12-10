import sklearn.metrics as M
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys

def get_results(npzfile):
    with np.load(npzfile) as data:
        logits = data['logits']
        preds = data['pred']
        labels = data['true']
    return logits,preds,labels

def report(preds,labels):
    print(M.classification_report(labels,preds))

def ROC(logits,labels,class_idx):
    probs = tf.nn.softmax(logits,axis=-1).numpy()[:,class_idx]
    fpr,tpr,threshold = M.roc_curve(labels,probs)
    auc = M.auc(fpr,tpr)

    plt.plot(fpr,tpr,'b',label='AUC = {:0.2f}'.format(auc))
    plt.plot([0,0],[1,1],'r--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    plt.legend()
    plt.title("ROC")
    plt.show()
    pass

def main():
    assert len(sys.argv) == 2,"Requires path to test results .npz file as argument"

    npzfile = sys.argv[1]

    logits,preds,labels = get_results(npzfile)

    report(preds,labels)
    ROC(logits,labels,1)
    pass

if __name__ == "__main__":
    main()
