import re
import csv
import sys
import random
import json
import numpy as np
import gensim
from gensim.models.keyedvectors import KeyedVectors
import nltk
nltk.download('punkt')

def short_to_decimal1(code):
    """
    Convert an ICD9 code from short format to decimal format.
    """
    if len(code) > 2:
        return code[:2] + '.' + code[2:]
    else:
        return code


def _zero_pad(x, n=2):
    if len(x) < n:
        x = (n - len(x)) * "0" + x
    return x

def decimal_to_short(code):
    """
    Convert an ICD9 code from decimal format to short format.
    """

    parts = code.split(".")
    parts[0] = _zero_pad(parts[0])

    return "".join(parts)

def short_to_decimal(code):
    """
    Convert an ICD9 code from short format to decimal format.
    """
    if 'V' not in code and 'E' not in code:
        if len(code) > 3:
            return code[:3] + '.' + code[3:]
        else:
            return code
    elif 'V' in code:
        if len(code) > 3:
            return code[:3] + '.' + code[3:]
        else:
            return code
    else:
        if len(code) > 4:
            return code[:4] + '.' + code[4:]
        else:
            return code


def load_descriptions_mim2():
    lookup = {}
    tmp_desc = {}
    with open('/home/amri228/final_paper/data/mimic2/ICD9_descriptions', 'r') as in_file:
        for row in in_file:
            data = row.strip().split('\t')
            tmp_desc[data[0]] = data[1]

    with open('/home/amri228/final_paper/data/mimic2/lookup_ids_labels.txt', 'r') as in_file:
        for row in in_file:
            data = row.strip().split('|')
            lookup[data[0]] = data[1]
    descs = []
    with open('/home/amri228/final_paper/data/mimic2/all_labels_final.txt', 'r') as in_file:
        for row in in_file:
            data = row.strip()
            #if lookup[data] in tmp_desc:
            descs.append(tmp_desc[lookup[data]])
            #else:
            #    descs.append('unknownabc')
    return descs



def load_descriptions():
    lookup = {}
    tmp_desc = {}
    missing = 0
    with open('/home/amri228/final_paper/data/mimic2/ICD9_descriptions', 'r') as in_file:
        for row in in_file:
            data = row.strip().split('\t')
            tmp_desc[data[0]] = data[1]
            if data[0] == '53.83':
                print decimal_to_short(data[0])
            #lookup[decimal_to_short(data[0])] = data[0]
            lookup[data[0]] = data[0]
    with open('/home/amri228/final_paper/data/icd9.txt', 'r') as in_file:
        for row in in_file:
            data = row.strip().split('\t')
            if data[0] in lookup:
                continue
            tmp_desc[short_to_decimal(data[0])] = data[1]
            lookup[short_to_decimal(data[0])] = short_to_decimal(data[0])
    with open('/home/amri228/final_paper/data/mimic3/all_data/D_ICD_PROCEDURES.csv', 'r') as in_file:
        iCSV = csv.reader(in_file, delimiter=',')
        for row in iCSV:
            data = []
            data.append(short_to_decimal1(row[1]))
            data.append(row[3])
            if data[0] in lookup:
                continue
            if row[1] == '400':
                print data
            tmp_desc[short_to_decimal1(data[0])] = data[1]
            lookup[short_to_decimal1(data[0])] = short_to_decimal1(data[0])
    with open('/home/amri228/final_paper/data/mimic3/all_data/D_ICD_DIAGNOSES.csv', 'r') as in_file:
        iCSV = csv.reader(in_file, delimiter=',')
        for row in iCSV:
            data = []
            data.append(short_to_decimal(row[1]))
            data.append(row[3])
            if data[0] in lookup:
                continue
            tmp_desc[short_to_decimal(data[0])] = data[1]
            lookup[short_to_decimal(data[0])] = short_to_decimal(data[0])

    with open('/home/amri228/final_paper/data/concept_embeddings/eval/cui_icd9.txt', 'r') as in_file:
        for row in in_file:
            data = row.strip().split('|')
            if data[0] in lookup:
                continue
            missing += 1
            tmp_desc[data[10]] = data[14]
            #lookup[decimal_to_short(data[10])] = data[10]
            lookup[data[10]] = data[10]

    descs = []
    #with open('/home/amri228/final_paper/data/mimic2/all_labels_final.txt', 'r') as in_file:
    missing = []
    with open('/home/amri228/final_paper/data/mimic3/fixed2_all_labels_final.txt', 'r') as in_file:
        for row in in_file:
            data = row.strip()
            if data in lookup:
                #descs.append(tmp_desc[lookup[data]])
                descs.append(tmp_desc[data])
            else:
                missing.append(data)
                descs.append('UNK')
    print 'total missing', len(missing), missing
    sys.stdout.flush()
    return descs

def load_data_file(txt_filename):
    txt = open(txt_filename, 'r')
    X_txt = []
    Y = []
    for row in txt:
        data = json.loads(row.strip())
        #X_txt.append(' '.join(nltk.word_tokenize(data['text'])))
        if 'txt' in data:
            X_txt.append(data['txt'])
        else:
            X_txt.append(data['text'])
        Y.append([x for x in data['labels'] if x != ''])
    txt.close()
    return X_txt, Y

class ProcessData(object):
    def __init__(self, pretrain_wv=None, lower=True, min_df=5, nltk=False):
        self.pattern = re.compile(r'(?u)\b\w\w+\b')
        #self.pattern = re.compile('[A-Z][a-z]+')
        self.nltk = nltk
        self.min_df = min_df
        self.lower = lower
        if pretrain_wv is not None:
            #self.wv = gensim.models.Word2Vec.load(pretrain_wv)
            self.wv = KeyedVectors.load_word2vec_format('/home/amri228/chemprot/data2/glove/glove_300d_w2v_format.txt', binary=False)
        else:
            self.wv = None
        self.embs = [np.zeros((300,)),
            np.random.uniform(-1.,1., (300,))*0.01]
        self.word_index = {None:0, 'UNK':1}

    def _tokenize(self, string):
        if self.lower:
            example = string.strip().lower()
        else:
            example = string.strip().lower()
        if self.nltk and False:
            return nltk.word_tokenize(example)
        else:
            return re.findall(self.pattern, example)

    def fit(self, data):
        token_cnts = {}
        excnt = 1
        for ex in data:
            #print excnt 
            excnt += 1
            example_tokens = self._tokenize(ex)
            for token in example_tokens:
                if token not in token_cnts:
                    token_cnts[token] = 1
                else:
                    token_cnts[token] += 1

        index = 2
        for value, key in enumerate(token_cnts):
            if value < self.min_df:
                continue
            self.word_index[key] = index
            if self.wv is not None:
                if key in self.wv:
                    self.embs.append(self.wv[key])
                else:
                    self.embs.append(np.random.uniform(-1.,1., (300,))*0.01)
            else:
                #self.embs.append(np.random.random((300,))*0.01)
                self.embs.append(np.random.uniform(-1.,1., (300,))*0.01)
            index += 1

        self.embs = np.array(self.embs)
        del self.wv
        return

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def transform(self, data):
        return_dataset = []
        for ex in data:
            example = self._tokenize(ex)
            index_example = []
            for token in example:
                if token in self.word_index:
                    index_example.append(self.word_index[token])
                else:
                    index_example.append(self.word_index['UNK'])
            return_dataset.append(index_example)

        return return_dataset

    def pad_data(self, data, to_shuffle=False):
        max_len = np.max([len(x) for x in data]) + 5
        padded_dataset = []
        for ex in data:
            if to_shuffle:
                #example = random.sample(ex, len(ex))
                example = ex
            else:
                example = ex
            zeros = [0]*(max_len-len(example))
            padded_dataset.append(example+zeros)
        return np.array(padded_dataset)

    def pad_data_hier(self, data):
        max_sents = np.max([len(x) for x in data])
        max_len = np.max([len(x) for y in data for x in y])
        padded_dataset = []
        for par in data:
            pad_sents = []
            for example in par:
                zeros = [0]*(max_len-len(example))
                pad_sents.append(example+zeros)
            for x in range(max_sents-len(par)):
                zeros = [0]*max_len
                pad_sents.append(zeros)
            padded_dataset.append(pad_sents)
        return np.array(padded_dataset)

class ProcessHierData(object):
    def __init__(self, pretrain_wv=None, lower=True, min_df=5):
        self.pattern = re.compile(r'(?u)\b\w\w+\b')
        self.min_df = min_df
        self.lower = lower
        if pretrain_wv is not None:
            self.wv = gensim.models.Word2Vec.load(pretrain_wv)
        else:
            self.wv = None
        self.embs = [np.zeros((300,)),
            np.random.random((300,))*0.01]
        self.word_index = {None:0, 'UNK':1}

    def _tokenize(self, string):
        if self.lower:
            example = string.strip().lower()
        else:
            example = string.strip()
        return re.findall(self.pattern, example)

    def fit(self, data):
        token_cnts = {}
        for par in data:
            sent_text = nltk.sent_tokenize(par)
            for ex in sent_text:
                example_tokens = self._tokenize(ex)
                for token in example_tokens:
                    if token not in token_cnts:
                        token_cnts[token] = 1
                    else:
                        token_cnts[token] += 1

        index = 2
        for value, key in enumerate(token_cnts):
            if value < self.min_df:
                continue
            self.word_index[key] = index
            if self.wv is not None:
                if key in self.wv:
                    self.embs.append(self.wv[key])
                else:
                    self.embs.append(np.random.random((300,))*0.01)
            else:
                self.embs.append(np.random.random((300,))*0.01)
            index += 1

        self.embs = np.array(self.embs)
        del self.wv
        return

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def transform(self, data):
        return_dataset = []
        for par  in data:
            sent_text = nltk.sent_tokenize(par)
            index_sents = []
            for ex in sent_text:
                example = self._tokenize(ex)
                index_example = []
                for token in example:
                    if token in self.word_index:
                        index_example.append(self.word_index[token])
                    else:
                        index_example.append(self.word_index['UNK'])
                index_sents.append(index_example)
            return_dataset.append(index_sents)
        return return_dataset

    def pad_data(self, data):
        max_sents = np.max([len(x) for x in data])
        max_len = np.max([len(x) for y in data for x in y])
        padded_dataset = []
        for par in data:
            pad_sents = []
            for example in par:
                zeros = [0]*(max_len-len(example))
                pad_sents.append(example+zeros)
            for x in range(max_sents-len(par)):
                zeros = [0]*max_len
                pad_sents.append(zeros)
            padded_dataset.append(pad_sents)
        return np.array(padded_dataset)
