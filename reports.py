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

def report(preds,labels,as_dict=False):
    return M.classification_report(labels,preds,output_dict=as_dict)

def ROC(logits,labels,class_idx,filename=None):
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
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    pass

def main():
    models = ["CNN","SENET","RESNET","SERESNET","RESNEXT","SERESNEXT"]
    losses = ["cross_entropy","F1","mean_F1","focal"]
    
    for m in models:
        for l in losses:
            try:
                folder = 'log/'+m+'/'+l+'/'
                npzfile = folder+'test_results.npz'
                imagename = folder+'roc.png'

                logits,preds,labels = get_results(npzfile)
                report = report(preds,labels)
                ROC(logits,labels,1,filename=imagename)

                with open('classification_report.txt','a+') as f:
                    f.write('{} with {} loss\n{}\n\n'.format(m,l,report))
                    
            except Exception as e:
                print("Failed on model '{}' with loss '{}' with exception\n{}".format(m,l,e))
    pass

if __name__ == "__main__":
    main()
