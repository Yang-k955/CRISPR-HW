import sys
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


fpath = './datasets/CIRCLE_seq_10gRNA_wholeDataset.csv'
dfGuideSeq = pd.read_csv(fpath, sep=',')
dfGuideSeq = dfGuideSeq[dfGuideSeq['label'] == 1]
dfGuideSeq = dfGuideSeq.reset_index(drop=True)

target_dna_list = []
target_rna_list = []

for n in range(len(dfGuideSeq)):
    target_dna = list(dfGuideSeq.loc[n, 'off_seq'])
    target_rna = list(dfGuideSeq.loc[n, 'sgRNA_seq'])

    if dfGuideSeq.loc[n, 'label'] == 1:
        for i in range(len(target_rna)):
            if target_rna[i] == 'N':
                target_rna[i] = target_dna[i]

        for i in range(len(target_dna)):
            if target_dna[i] >= 'a' and target_dna[i] <= 'z':
                target_dna[i] = chr(ord(target_dna[i]) - ord('a') + ord('A'))
            if target_dna[i] == 'N':
                target_dna[i] = target_rna[i]

        target_dna = ''.join(target_dna)
        target_rna = ''.join(target_rna)
        target_rna_list.append(target_rna)
        target_dna_list.append(target_dna)

arr = np.zeros((len(target_dna_list),22))
new_pd = pd.DataFrame(arr,columns=range(1,23))

for i in range(len(target_rna_list)):
    for j in range(len(target_rna_list[0])-2):
        a = target_rna_list[i][j]
        b = target_dna_list[i][j]
        if a == '_':
            a = '-'
        if b == '_':
            b = '-'
        if a != b or (a=='-' and b=='-'):
            piars = a + b
            if '-' in piars:
                new_pd.iat[i,j] = 1

    new_pd.iat[i,21] = dfGuideSeq.loc[i, 'Read']

cor = new_pd.corr()

aaa = cor[22][0:-1]
print(aaa)
plt.bar(range(1,22), cor[22][0:-1],color='#41D3BD')

#plt.title('(a) The effect of bulge location and type')
plt.xlabel('Bulge Position')
plt.ylabel('Effect on CIRCLE reads')
# plt.legend()

plt.xticks(range(1,22))
#plt.savefig('indel_position_bar.jpg')

plt.show()







