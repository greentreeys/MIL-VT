from sklearn import metrics
import numpy as np

def MultiClassLabelTransfer(LabelArray_MultiLabel, ProbArray_MultiLabel):


    LabelArray = np.array([list(x) for x in LabelArray_MultiLabel])
    ProbArray = np.array([list(x) for x in ProbArray_MultiLabel])
    
    PredArray_MultiClass = np.argmax(ProbArray, axis=1)

    LabelArray_MultiClass = []
    for index in range(len(LabelArray)):
        tempLabels = np.array(np.where(LabelArray[index, :]>0)[0])
        if len(tempLabels) == 1:
            LabelArray_MultiClass.append(tempLabels[0])
        else:
            tempPred = ProbArray[index, tempLabels]
            tempLabel2 = tempLabels[np.argmax(tempPred)]
            LabelArray_MultiClass.append(tempLabel2)

    return LabelArray_MultiClass, PredArray_MultiClass

def getRecall_MultiClass(LabelArray_MultiClass, PredArray_MultiClass):
    recall = metrics.recall_score(LabelArray_MultiClass, PredArray_MultiClass, average=None)
    avgRecall = np.mean(recall)
    return avgRecall, recall

def getPrecision_MultiClass(LabelArray_MultiClass, PredArray_MultiClass):
    precision = metrics.precision_score(LabelArray_MultiClass, PredArray_MultiClass, average=None)
    avgPrecision = np.mean(precision)
    return avgPrecision, precision

def getF1_MultiClass(LabelArray_MultiClass, PredArray_MultiClass):
    f1 = metrics.f1_score(LabelArray_MultiClass, PredArray_MultiClass, average=None)
    avgF1 = np.mean(f1)
    return avgF1, f1

def getAUC_MultiLabel(LabelArray_MultiLabel, ProbArray_MultiLabel):
    LabelArray_MultiLabel = np.array([list(x) for x in LabelArray_MultiLabel])
    ProbArray_MultiLabel = np.array([list(x) for x in ProbArray_MultiLabel])

    auc = np.zeros(ProbArray_MultiLabel.shape[1])
    for i in range(ProbArray_MultiLabel.shape[1]):
        auc[i] = metrics.roc_auc_score(LabelArray_MultiLabel[:,i], ProbArray_MultiLabel[:,i])
    avgAUC = np.mean(auc)

    return avgAUC, auc
