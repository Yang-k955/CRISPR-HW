import numpy as np

#crispr-ip
def crispr_ip_coding(target_seq, off_target_seq):
    encoded_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1], '_': [0, 0, 0, 0],
                    '-': [0, 0, 0, 0]}
    pos_dict = {'A': 1, 'T': 2, 'G': 3, 'C': 4, '_': 5, '-': 5}
    tlen = 24
    target_seq = "-" * (tlen - len(target_seq)) + target_seq.upper()
    off_target_seq = "-" * (tlen - len(off_target_seq)) + off_target_seq.upper()

    target_seq_code = np.array([encoded_dict[base] for base in list(target_seq)])
    off_target_seq_code = np.array([encoded_dict[base] for base in list(off_target_seq)])
    on_off_dim6_codes = []
    for i in range(len(target_seq)):
        diff_code = np.bitwise_or(target_seq_code[i], off_target_seq_code[i])
        dir_code = np.zeros(2)
        if pos_dict[target_seq[i]] == pos_dict[off_target_seq[i]]:
            diff_code = diff_code * -1
            dir_code[0] = 1
            dir_code[1] = 1
        elif pos_dict[target_seq[i]] < pos_dict[off_target_seq[i]]:
            dir_code[0] = 1
        elif pos_dict[target_seq[i]] > pos_dict[off_target_seq[i]]:
            dir_code[1] = 1
        else:
            raise Exception("Invalid seq!", target_seq, off_target_seq)
        on_off_dim6_codes.append(np.concatenate((diff_code, dir_code)))
    on_off_dim6_codes = np.array(on_off_dim6_codes)
    isPAM = np.zeros((24, 1))
    isPAM[-3:, :] = 1
    on_off_code = np.concatenate((on_off_dim6_codes, isPAM), axis=1)
    return on_off_code

class Encoder():
    def __init__(self, on_seq, off_seq):
        tlen = 24
        self.on_seq = "-" *(tlen-len(on_seq)) +  on_seq.upper()
        self.off_seq = "-" *(tlen-len(off_seq)) + off_seq.upper()
        self.encoded_dict_indel = {'A': [1, 0, 0, 0, 0], 'T': [0, 1, 0, 0, 0],
                                   'G': [0, 0, 1, 0, 0], 'C': [0, 0, 0, 1, 0], '_': [0, 0, 0, 0, 1], '-': [0, 0, 0, 0, 0]}
        self.direction_dict = {'A':5, 'G':4, 'C':3, 'T':2, '_':1}
        self.encode_on_off_dim7()

    def encode_sgRNA(self):
        code_list = []
        encoded_dict = self.encoded_dict_indel
        sgRNA_bases = list(self.on_seq)
        for i in range(len(sgRNA_bases)):
            if sgRNA_bases[i] == "N":
                sgRNA_bases[i] = list(self.off_seq)[i]
            code_list.append(encoded_dict[sgRNA_bases[i]])
        self.sgRNA_code = np.array(code_list)

    def encode_off(self):
        code_list = []
        encoded_dict = self.encoded_dict_indel
        off_bases = list(self.off_seq)
        for i in range(len(off_bases)):
            code_list.append(encoded_dict[off_bases[i]])
        self.off_code = np.array(code_list)

    def encode_on_off_dim7(self):
        self.encode_sgRNA()
        self.encode_off()
        on_bases = list(self.on_seq)
        off_bases = list(self.off_seq)
        on_off_dim7_codes = []
        for i in range(len(on_bases)):
            diff_code = np.bitwise_or(self.sgRNA_code[i], self.off_code[i])
            on_b = on_bases[i]
            off_b = off_bases[i]
            if on_b == "N":
                on_b = off_b
            dir_code = np.zeros(2)
            if on_b == "-" or off_b == "-" or self.direction_dict[on_b] == self.direction_dict[off_b]:
                pass
            else:
                if self.direction_dict[on_b] > self.direction_dict[off_b]:
                    dir_code[0] = 1
                else:
                    dir_code[1] = 1
            on_off_dim7_codes.append(np.concatenate((diff_code, dir_code)))
        self.on_off_code = np.array(on_off_dim7_codes)



def cnn_predict(guide_seq, off_seq):
    code_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}
    gRNA_list = list(guide_seq)
    off_list = list(off_seq)
    pair_code = []
    for i in range(len(gRNA_list)):
        if gRNA_list[i] == 'N':
            gRNA_list[i] = off_list[i]
        gRNA_base_code = code_dict[gRNA_list[i]]
        DNA_based_code = code_dict[off_list[i]]
        pair_code.append(list(np.bitwise_or(gRNA_base_code, DNA_based_code)))
    input_code = np.array(pair_code).reshape(1, 1, 23, 4)
    return input_code


def crispr_net_coding(on_seq,off_seq):
    e = Encoder(on_seq=on_seq, off_seq=off_seq)
    return e.on_off_code



def CNN_std_coding(guide_seq, off_seq):
    code_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}
    gRNA_list = list(guide_seq)
    off_list = list(off_seq)
    if len(gRNA_list) != len(off_list):
        return 0
    pair_code = []
    for i in range(1,len(gRNA_list)):
        if gRNA_list[i] == 'N':
            gRNA_list[i] = off_list[i]
        gRNA_base_code = code_dict[gRNA_list[i]]
        DNA_based_code = code_dict[off_list[i]]
        pair_code.append(list(np.bitwise_or(gRNA_base_code, DNA_based_code)))
    return pair_code

