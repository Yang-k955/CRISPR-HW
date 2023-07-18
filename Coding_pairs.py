
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

encoding_map = {
    'AA': 0, 'AC': 1, 'AG': 2, 'AT': 3,
    'CA': 4, 'CC': 5, 'CG': 6, 'CT': 7,
    'GA': 8, 'GC': 9, 'GG': 10, 'GT': 11,
    'TA': 12, 'TC': 13, 'TG': 14, 'TT': 15,
    '-A': 16, '-C': 17,'-G': 18, '-T': 19,
    'A-': 20, 'C-': 21,'G-': 22, 'T-': 23,
    '--': 24
}

def encode_sequence(sgRNA, DNA):
    pairs = [(sgRNA[i].upper() if sgRNA[i] != '_' else '-') + (DNA[i].upper() if DNA[i] != '_' else '-') for i in range(len(sgRNA))]
    return [encoding_map[p] for p in pairs]

def Clean_data(filepath,filename):
    tlen = 24
    data = pd.read_csv(filepath+filename+".csv")
    on_seqs = data["on_seq"]
    off_seqs = data["off_seq"]
    labels = data["label"]
    on_seqs = on_seqs.apply(lambda on_seq: "-" * (tlen - len(on_seq)) + on_seq)
    off_seqs = off_seqs.apply(lambda off_seq: "-" * (tlen - len(off_seq)) + off_seq)
    labels = labels.apply(lambda label: int(label != 0))

    sgRNAs_new = []
    for index, sgRNA in enumerate(on_seqs):
        sgRNA = list(sgRNA)
        sgRNA[-3] = off_seqs[index][-3]
        sgRNAs_new.append(''.join(sgRNA))
    on_seqs = pd.Series(sgRNAs_new)
    data = pd.DataFrame.from_dict({'on_seqs': on_seqs, 'off_seqs': off_seqs, 'labels': labels})
    return data[data.apply(lambda row: 'N' not in list(row['off_seqs']), axis=1)]


if __name__ == '__main__':
    '''
    File = {"./Dataset_M/": ["CRISPOR", "Doench", "GUIDE-Seq_Kleinstiver", "GUIDE-Seq_Listgarten",
                             "GUIDE-Seq_Tasi", "SITE-Seq","Hek293t","K562","Hek293t_K562","deepCrispr_OT_data"],
            "./Dataset_IM/": ["CIRCLE_seq", "Listgarten"]}
    '''
    File = {"./Dataset_M/": ["K562"]}
    for filepath in File:
        for filename in File[filepath]:
            data = Clean_data(filepath, filename)
            print(f"coding.....{filepath}{filename}")
            encoded_data = data.apply(lambda row: encode_sequence(row['on_seqs'], row['off_seqs']), axis=1)
            print("end coding.....")
            data['encoding'] = encoded_data.apply(lambda x: ','.join(map(str, x)))
            train, test = train_test_split(data, test_size=0.2, random_state=42)
            file_prefix = filename.split('.')[0]

            print("saving.....")
            with open(f'./coding/Train-{file_prefix}.txt', 'w') as f:
                for index, row in train.iterrows():
                    f.write(f"{row['on_seqs']},{row['off_seqs']},{row['labels']},{row['encoding']}\n")
                    print(f"{index}:  {row['on_seqs']},{row['off_seqs']},{row['encoding']}")
            with open(f'./coding/Test-{file_prefix}.txt', 'w') as f:
                for index, row in test.iterrows():
                    f.write(f"{row['on_seqs']},{row['off_seqs']},{row['labels']},{row['encoding']}\n")
                    print(f"{index}:  {row['on_seqs']},{row['off_seqs']},{row['encoding']}")
            print("end.....")

