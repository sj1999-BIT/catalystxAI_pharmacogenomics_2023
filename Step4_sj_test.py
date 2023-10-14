# python3
# -*- coding:utf-8 -*-

"""
@author:野山羊骑士
@e-mail：thankyoulaojiang@163.com
@file：PycharmProject-PyCharm-model.py
@time:2021/9/15 16:33 
"""

import os
import numpy as np
import pandas as pd
import codecs
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
from scipy.stats import pearsonr,spearmanr
import copy
import time
import pickle

import torch
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import SequentialSampler

from prettytable import PrettyTable
from subword_nmt.apply_bpe import BPE
from model_helper import Encoder_MultipleLayers, Embeddings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from Step2_DataEncoding import DataEncoding
from Step3_model import *


if __name__ == '__main__':

    modeldir = '.'
    modelfile = modeldir + '/model.pt'
    if not os.path.exists(modeldir):
        print("no model!")

    vocab_dir = '.'
    obj = DataEncoding(vocab_dir=vocab_dir)
    # 切分完成
    traindata, testdata = obj.Getdata.ByCancer(random_seed=1)
    # encoding 完成
    traindata, train_rnadata, testdata, test_rnadata = obj.encode(
        traindata=traindata,
        testdata=testdata)

    net = DeepTTC(modelfile)
    net.load_pretrained(modelfile)
    y_label, y_pred, mse, rmse, person, p_val, spearman, s_p_val, CI = \
        net.predict(drug_data=testdata, rna_data=test_rnadata)
    print(f'y_label: {y_label}')
    print(f'y_pred: {y_pred}')
    print(f'mse: {mse}')
    print(f'rmse: {rmse}')
    print(f'person: {person}')
    print(f'p_val: {p_val}')
    print(f'spearman: {spearman}')
    print(f's_p_val: {s_p_val}')
    print(f'CI: {CI}')
    # net.save_model()
    # print("Model Saveed :{}".format(modelfile))



