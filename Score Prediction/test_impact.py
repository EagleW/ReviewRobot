import os
import sys
import json
import time
import torch
import pickle
import argparse
import numpy as np
from torch import nn
from loader.logger import Tee
from collections import OrderedDict
from model.RNN_torch import Impact
from sklearn.metrics import accuracy_score, mean_squared_error
from loader.utils import create_impact
from loader.data_loader import load_impact
from loader.dataset import Impact_Dataset, Impact_Processor

parser = argparse.ArgumentParser()
parser.add_argument(
    "--type", default="0",
    type=int, help="0 for acl, 1 for iclr"
)
parser.add_argument(
    "--word_dim", default="64",
    type=int, help="Token embedding dimension"
)
parser.add_argument(
    "--feature_dim", default="1",
    type=int, help="Feature embedding dimension"
)
parser.add_argument(
    "--hidden_dim", default="32",
    type=int, help="Hidden dimension"
)
parser.add_argument(
    "--model_dp", default="models/impact_models/",
    help="model directory path"
)
parser.add_argument(
    "--gpu", default="1",
    type=int, help="default is 1. set 0 to disable use gpu."
)
parser.add_argument(
    "--batch_size", default="32",
    type=int, help="Batch size."
)
parser.add_argument(
    "--dropout", default="0.1",
    type=float, help="Dropout on the embeddings (0 = no dropout)"
)
parser.add_argument(
    "--layer_dropout", default="0.2",
    type=float, help="Dropout on the embeddings (0 = no dropout)"
)
parser.add_argument(
    "--max_len", default="200",
    type=int, help="Max length."
)
parser.add_argument(
    "--freq", default="5",
    type=int, help="Min freq."
)
parser.add_argument(
    "--lr_rate", default="0.0005",
    type=float, help="Learning rate"
)
parser.add_argument(
    "--data_path", default="data",
    help="data directory path"
)
parser.add_argument(
    "--load", action='store_true', help="Load dataset."
)
parser.add_argument(
    "--num_epochs", default="100",
    type=int, help="Number of training epochs"
)                
parser.add_argument(
    "--model", default="best_dev_model.pth.tar",
    help="Model location"
)
args = parser.parse_args()

parameters = OrderedDict()
parameters['freq'] = args.freq
parameters['w_dim'] = args.word_dim
parameters['f_dim'] = args.feature_dim
parameters['h_dim'] = args.hidden_dim
parameters['input_dropout'] = args.dropout
parameters['layer_dropout'] = args.layer_dropout
parameters['gpu'] = args.gpu == 1
parameters['batch_size'] = args.batch_size
parameters['max_len'] = args.max_len
parameters['lr_rate'] = args.lr_rate
parameters['num_class'] = 5

state = pickle.load(open('data/impact_dataset.pth', 'rb'))
mappings = state['mappings']
word2id = mappings['word2id']
id2word = mappings['id2word']
vocab_size = len(mappings['id2word'])
t_dataset = state['t_dataset']

f1_bins = mappings['f1']
f1_size = len(f1_bins) + 1
test_dataset = Impact_Dataset(t_dataset, word2id, f1_bins, parameters['max_len'])
apro_proc = Impact_Processor()


device = torch.device("cuda:1" if torch.cuda.is_available() and parameters['gpu'] else "cpu")


embed_layer_word = nn.Embedding(vocab_size, args.word_dim, padding_idx=0, sparse=False)
embed_layer_f1 = nn.Embedding(f1_size, args.feature_dim, padding_idx=0, sparse=False)

model = Impact(embed_layer_word, embed_layer_f1, **parameters)
model = model.to(device)


state = torch.load('models/impact_models/best/best_dev_model.pth.tar')
state_dict = state['model']
model.load_state_dict(state_dict)

model.eval()
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=False,
    collate_fn=apro_proc.process
)
aspect_all_ys = []
aspect_all_ys_ = []
aspect_all_ys_avg = []
for batch_idx, (batch_txts, batch_lens, batch_f1s, batch_tgt, batch_ids, batch_avg) in enumerate(test_loader):
    batch_txts =batch_txts.to(device)
    batch_f1s = batch_f1s.to(device)
    batch_tgt = batch_tgt.to(device)
    prob = model(batch_txts, batch_lens, batch_f1s)
    symbols = prob.topk(1)[1].squeeze(1)
    aspect_all_ys_.extend(symbols.tolist())
    aspect_all_ys.extend(batch_tgt.tolist())
    aspect_all_ys_avg.extend(batch_avg)
r = accuracy_score(aspect_all_ys, aspect_all_ys_)
m = mean_squared_error(aspect_all_ys, aspect_all_ys_)
print()
print('\t accuracy %.4f' % (r))
print('\t mse %.4f' % (m))
decision_accuracy = 0
for ys_, ys in zip(aspect_all_ys_, aspect_all_ys_avg):
    if (ys_+1 > 3 and ys > 3.5) or (ys_+1 <= 3 and ys <= 3.5):
        decision_accuracy += 1
print('\t decision accuracy', decision_accuracy/len(aspect_all_ys_))
print(aspect_all_ys, aspect_all_ys_, aspect_all_ys_avg)