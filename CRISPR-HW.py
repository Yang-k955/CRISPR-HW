# -*- coding: UTF-8 -*-
'''
@Project ：mymodel
@File ：CRISPR-HW.py
@IDE  ：PyCharm
@Author ：'YANG YANPENG'
@Date ：2023/4/9 15:34
'''

from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from model.model import CRISPR_HW, CRISPR_HW_noblstm, CRISPR_HW_noresnet, CRISPR_HW_noatt, CRISPR_HW_Linear, CRISPR_HW_nodense
import keras
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score, confusion_matrix,roc_curve, precision_recall_curve
import os
from keras.callbacks import ReduceLROnPlateau


def loadData(encoded_file):
    data = pd.read_csv(encoded_file, header=None)
    return data

"""def confusion_matrix():
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()"""


if __name__ == '__main__':
    num_classes = 2
    epochs = 30
    batch_size = 128 #512,1024,1000 #Different batch_size are required for different datasets
    lr = 0.003
    #FileList = ["SITE-Seq", "CIRCLE_seq","GUIDE-Seq_Kleinstiver", "CRISPOR", "GUIDE-Seq_Listgarten", "GUIDE-Seq_Tasi","deepCrispr_OT_data","Doench","Hek293t", "K562","Listgarten","Hek293t_K562"]
    FileList = ["CRISPOR", "Doench", ]
    for dataname in FileList:

        results = pd.DataFrame(
            columns=['model','dataset','Fold', 'Accuracy', 'F1_score', 'Precision', 'Recall', 'ROC_AUC', 'PR_AUC'])

        encoded_file = f"datasets/word embedding/{dataname}.txt"
        data = np.array(loadData(encoded_file))
        xdata = data[:, 3:]
        ydata = data[:, 2]
        xdata = xdata.astype('int')
        ydata = ydata.astype('int')

        ydata = to_categorical(ydata, num_classes)
        train_data_encodings, train_labels = shuffle(xdata, ydata, random_state=2023)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
        # [CRISPR_HW, CRISPR_HW_noblstm, CRISPR_HW_noresnet, CRISPR_HW_noatt, CRISPR_HW_Linear]
        for modelname in [CRISPR_HW]:
            print(f"start train {dataname} on {modelname.__name__}")
            fold = 1
            for train_index, test_index in skf.split(xdata, np.argmax(ydata, axis=1)):
                xtrain, xtest = xdata[train_index], xdata[test_index]
                ytrain, ytest = ydata[train_index], ydata[test_index]
                model = modelname()

                model.compile(loss=keras.losses.CategoricalCrossentropy(),
                              optimizer=keras.optimizers.Adam(lr=lr, amsgrad=False),metrics=['accuracy'])

                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5, verbose=1)

                history = model.fit(xtrain, ytrain, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[reduce_lr])

                y_pred_probs = model.predict(xtest)
                y_pred = np.argmax(y_pred_probs, axis=1)
                y_true = np.argmax(ytest, axis=1)
                y_score = y_pred_probs[:, 1]

                acc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                roc_auc = roc_auc_score(y_true, y_score)
                pr_auc = average_precision_score(y_true, y_score)
                acc = round(acc, 4)
                f1 = round(f1, 4)
                precision = round(precision, 4)
                recall = round(recall, 4)
                roc_auc = round(roc_auc, 4)
                pr_auc = round(pr_auc, 4)

                results = results.append(
                    {'model': modelname.__name__,'dataset':dataname , 'Fold': fold, 'Accuracy': acc, 'F1_score': f1, 'Precision': precision, 'Recall': recall,
                     'ROC_AUC': roc_auc, 'PR_AUC': pr_auc }, ignore_index=True)
                fold += 1

            recent_results = results.tail(5)
            accuracy_mean = round(recent_results['Accuracy'].mean(),4)
            f1_score_mean = round(recent_results['F1_score'].mean(),4)
            precision_mean = round(recent_results['Precision'].mean(),4)
            recall_mean = round(recent_results['Recall'].mean(),4)
            roc_auc_mean = round(recent_results['ROC_AUC'].mean(),4)
            pr_auc_mean = round(recent_results['PR_AUC'].mean(),4)
            average_row = {'model': modelname.__name__, 'dataset': dataname, 'Fold': 'All', 'Accuracy': accuracy_mean,
                           'F1_score': f1_score_mean,
                           'Precision': precision_mean, 'Recall': recall_mean, 'ROC_AUC': roc_auc_mean,
                           'PR_AUC': pr_auc_mean}
            results = results.append(average_row, ignore_index=True)

        results.to_csv('./results/results.csv', mode='a', index=False, header=False)
        print(results)

