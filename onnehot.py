import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

from datas.onehot.coding import crispr_ip_coding, crispr_net_coding
from model.model import crispr_HW_onehot
import keras
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score, confusion_matrix,roc_curve, precision_recall_curve
import os
from keras.callbacks import ReduceLROnPlateau


def loadData(encoded_file):
    data = pd.read_csv(encoded_file, header=None)
    return data


if __name__ == '__main__':
    num_classes = 2
    epochs = 30
    batch_size = 10000
    lr = 0.003
    # FileList = ["SITE-Seq", "CIRCLE_seq","GUIDE-Seq_Kleinstiver", "CRISPOR", "GUIDE-Seq_Listgarten", "GUIDE-Seq_Tasi","deepCrispr_OT_data","Doench","Hek293t", "K562","Listgarten","Hek293t_K562"]
    # FileList = ["SITE-Seq", "Hek293t", "K562", "Hek293t_K562"]
    FileList = ["CIRCLE_seq"]
    for dataname in FileList:
        results = pd.DataFrame(
            columns=['model', 'dataset', 'Fold', 'Accuracy', 'F1_score', 'Precision', 'Recall', 'ROC_AUC', 'PR_AUC'])
        print(f"start train {dataname}")
        print('Encoding!!')
        #加载数据
        encoded_file = f"datas/coding-all/{dataname}.txt"
        train_data = np.array(loadData(encoded_file))
        txdata = train_data[:, 0:2]
        tydata = train_data[:, 2]
        tydata = tydata.astype(int)
        # 转换为 Pandas DataFrame
        txdata = pd.DataFrame(txdata)

        train_data_encodings = np.array(
            txdata.apply(lambda row: crispr_net_coding(row[0], row[1]), axis=1).to_list()
        )
        train_data_encodings = train_data_encodings.reshape((len(train_data_encodings), 1, 24, 7))

        train_labels = to_categorical(tydata, num_classes)

        train_data_encodings, train_labels = shuffle(train_data_encodings, train_labels, random_state=2023)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
        fold = 1
        for train_index, test_index in skf.split(train_data_encodings, np.argmax(train_labels, axis=1)):
            xtrain, xtest = train_data_encodings[train_index], train_data_encodings[test_index]
            ytrain, ytest = train_labels[train_index], train_labels[test_index]
            model = crispr_HW_onehot()

            model.compile(loss=keras.losses.CategoricalCrossentropy(),
                          optimizer=keras.optimizers.Adam(lr=lr, amsgrad=False),metrics=['accuracy'])

            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5, verbose=1)

            history = model.fit(xtrain, ytrain, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[reduce_lr], validation_data=(xtest, ytest))

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
                {'model': 'onehot', 'dataset': dataname, 'Fold': fold, 'Accuracy': acc, 'F1_score': f1,
                 'Precision': precision, 'Recall': recall,
                 'ROC_AUC': roc_auc, 'PR_AUC': pr_auc}, ignore_index=True)
            fold += 1

        recent_results = results.tail(5)
        accuracy_mean = round(recent_results['Accuracy'].mean(), 4)
        f1_score_mean = round(recent_results['F1_score'].mean(), 4)
        precision_mean = round(recent_results['Precision'].mean(), 4)
        recall_mean = round(recent_results['Recall'].mean(), 4)
        roc_auc_mean = round(recent_results['ROC_AUC'].mean(), 4)
        pr_auc_mean = round(recent_results['PR_AUC'].mean(), 4)
        average_row = {'model': "onehot", 'dataset': dataname, 'Fold': 'All', 'Accuracy': accuracy_mean,
                       'F1_score': f1_score_mean,
                       'Precision': precision_mean, 'Recall': recall_mean, 'ROC_AUC': roc_auc_mean,
                       'PR_AUC': pr_auc_mean}
        results = results.append(average_row, ignore_index=True)
        results.to_csv('./results/results.csv', mode='a', index=False, header=False)
        print(results)

