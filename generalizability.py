import numpy as np
import pandas as pd
from model.model import CRISPR_HW
import keras
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score, confusion_matrix,roc_curve, precision_recall_curve
import os
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau


def loadData(encoded_file):
    data = pd.read_csv(encoded_file, header=None)
    return data



if __name__ == '__main__':
    num_classes = 2
    epochs = 30
    batch_size = 1000
    lr = 0.001
    tag = 5

    #FileList = ["SITE-Seq", "CIRCLE_seq", "CRISPOR", "Listgarten",  "Doench", "GUIDE-Seq_Kleinstiver", "GUIDE-Seq_Listgarten", "GUIDE-Seq_Tasi"]
    FileList = ["GUIDE-Seq_Kleinstiver", "GUIDE-Seq_Listgarten", "GUIDE-Seq_Tasi"]

    print(f"training data loading")
    data_list = []
    for dataname in FileList:
        encoded_file = f"./datasets/word embedding/{dataname}.txt"
        data_list.append(loadData(encoded_file))
    data = np.concatenate(data_list, axis=0)
    results = pd.DataFrame(
        columns=['model', 'Fold', 'Accuracy', 'F1_score', 'Precision', 'Recall', 'ROC_AUC', 'PR_AUC'])
    xdata = data[:, 3:]
    ydata = data[:, 2]
    xdata = xdata.astype('float32')
    ydata = ydata.astype('float32')
    ydata = to_categorical(ydata, num_classes)



    print(f"Testing data loading")
    test_list = []
    for dataname in ["CRISPOR",  "Doench"]:
        encoded_file = f"./datasets/word embedding/{dataname}.txt"
        test_list.append(loadData(encoded_file))
    test = np.concatenate(test_list, axis=0)

    xtest = test[:, 3:]
    ytest = test[:, 2]
    xtest = xtest.astype('float32')
    ytest = ytest.astype('float32')
    ytest = to_categorical(ytest, num_classes)

    print(f"starting training")
    model = CRISPR_HW()

    model.compile(loss=keras.losses.CategoricalCrossentropy(),
                  optimizer=keras.optimizers.Adam(lr=lr, amsgrad=False),metrics=['accuracy'])

    history = model.fit(xdata, ydata, epochs=epochs, batch_size=batch_size, verbose=1)

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

    print(f"Accuracy: {acc}")
    print(f"F1 score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"ROC AUC: {roc_auc}")
    print(f"PR AUC: {pr_auc}")

