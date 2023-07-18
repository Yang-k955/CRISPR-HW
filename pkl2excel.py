import os
import pandas as pd

def pkl2csv():
    folder_path = "./datasets/hmg_data (dataset II-2).pkl"
    data = pd.read_pickle(folder_path)
    data.to_csv('./datasets/hmg_data (dataset II-2).csv', index=False)

def sp():

    data = pd.read_csv('datases/elevation_6gRNA_wholeDataset.csv')
    #data['readFraction'] = data['readFraction'].apply(lambda x: 1 if x > 0 else x)
    data = data.loc[:, ['DNA', 'crRNA', 'label']]
    data = data.rename(columns={'DNA': 'on_seq', 'crRNA': 'off_seq', 'label': 'label'})
    data.to_csv('./datasets/GUIDE-Seq_Listgarten.csv', index=False)

if __name__ == '__main__':
    sp()