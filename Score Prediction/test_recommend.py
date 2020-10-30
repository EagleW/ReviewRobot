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
from model.RNN_torch import Recommend
from sklearn.metrics import accuracy_score, mean_squared_error
from loader.utils import create_recommend
from loader.data_loader import load_recommend
from loader.dataset import Recommend_Dataset, Recommend_Processor

parser = argparse.ArgumentParser()
parser.add_argument(
    "--type", default="0",
    type=int, help="0 for acl, 1 for iclr"
)
parser.add_argument(
    "--word_dim", default="128",
    type=int, help="Token embedding dimension"
)
parser.add_argument(
    "--feature_dim", default="4",
    type=int, help="Feature embedding dimension"
)
parser.add_argument(
    "--hidden_dim", default="64",
    type=int, help="Hidden dimension"
)
parser.add_argument(
    "--model_dp", default="models/recommend_models/",
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
    "--dropout", default="0.4",
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
    "--freq", default="1",
    type=int, help="Min freq."
)
parser.add_argument(
    "--lr_rate", default="0.0001",
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
    "--num_epochs", default="80",
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

state = pickle.load(open('data/recommend_dataset.pth', 'rb'))
mappings = state['mappings']
word2id = mappings['word2id']
id2word = mappings['id2word']
vocab_size = len(mappings['id2word'])

fa1_bins = mappings['fa1']
fa1_size = len(fa1_bins) + 1

fa2_bins = mappings['fa2']
fa2_size = len(fa2_bins) + 1

fr1_bins = mappings['fr1']
fr1_size = len(fr1_bins) + 1

fr2_bins = mappings['fr2']
fr2_size = len(fr2_bins) + 1

fs1_bins = mappings['fs1']
fs1_size = len(fs1_bins) + 1

fs2_bins = mappings['fs2']
fs2_size = len(fs2_bins) + 1

fn1_bins = mappings['fn1']
fn1_size = len(fn1_bins) + 1

fn2_bins = mappings['fn2']
fn2_size = len(fn2_bins) + 1

fi1_bins = mappings['fi1']
fi1_size = len(fi1_bins) + 1

t_dataset = state['t_dataset']
test_dataset = Recommend_Dataset(t_dataset, word2id, fa1_bins, fa2_bins, fs1_bins, fs2_bins, fn1_bins, fn2_bins, fr1_bins, fr2_bins, fi1_bins, parameters['max_len'])
apro_proc = Recommend_Processor()
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=False,
    collate_fn=apro_proc.process
)

embed_layer_word = nn.Embedding(vocab_size, args.word_dim, padding_idx=0, sparse=False)
embed_layer_fa1 = nn.Embedding(fa1_size, args.feature_dim, padding_idx=0, sparse=False)
embed_layer_fa2 = nn.Embedding(fa2_size, args.feature_dim, padding_idx=0, sparse=False)
embed_layer_fs1 = nn.Embedding(fs1_size, args.feature_dim, padding_idx=0, sparse=False)
embed_layer_fs2 = nn.Embedding(fs2_size, args.feature_dim, padding_idx=0, sparse=False)
embed_layer_fn1 = nn.Embedding(fn1_size, args.feature_dim, padding_idx=0, sparse=False)
embed_layer_fn2 = nn.Embedding(fn2_size, args.feature_dim, padding_idx=0, sparse=False)
embed_layer_fr1 = nn.Embedding(fr1_size, args.feature_dim, padding_idx=0, sparse=False)
embed_layer_fr2 = nn.Embedding(fr2_size, args.feature_dim, padding_idx=0, sparse=False)
embed_layer_fi1 = nn.Embedding(fi1_size, args.feature_dim, padding_idx=0, sparse=False)

model = Recommend(embed_layer_word, embed_layer_fa1, embed_layer_fa2, embed_layer_fs1, embed_layer_fs2, embed_layer_fn1, embed_layer_fn2, embed_layer_fr1, embed_layer_fr2, embed_layer_fi1, **parameters)
device = torch.device("cuda:1" if torch.cuda.is_available() and parameters['gpu'] else "cpu")
model = model.to(device)
state = torch.load('models/recommend_models/best/best_dev_model.pth.tar')
state_dict = state['model']
model.load_state_dict(state_dict)


model.eval()
aspect_all_ys = []
aspect_all_ys_ = []
aspect_all_ys_avg = []
aspect_ids = []
for batch_idx, (batch_txts, batch_lens, batch_fa1s, batch_fa2s, batch_fs1s, batch_fs2s, batch_fn1s, batch_fn2s, batch_fr1s, batch_fr2s, batch_fi1s, batch_tgt, batch_ids, batch_avg) in enumerate(test_loader):

    batch_txts =batch_txts.to(device)

    batch_fa1s = batch_fa1s.to(device)
    batch_fa2s = batch_fa2s.to(device)
    batch_fs1s = batch_fs1s.to(device)
    batch_fs2s = batch_fs2s.to(device)
    batch_fn1s = batch_fn1s.to(device)
    batch_fn2s = batch_fn2s.to(device)
    batch_fr1s = batch_fr1s.to(device)
    batch_fr2s = batch_fr2s.to(device)
    batch_fi1s = batch_fi1s.to(device)
    batch_tgt = batch_tgt.to(device)
    prob = model(batch_txts, batch_lens, batch_fa1s, batch_fa2s, batch_fs1s, batch_fs2s, batch_fn1s, batch_fn2s, batch_fr1s, batch_fr2s, batch_fi1s)
    symbols = prob.topk(1)[1].squeeze(1)
    aspect_all_ys_.extend(symbols.tolist())
    aspect_all_ys.extend(batch_tgt.tolist())
    aspect_all_ys_avg.extend(batch_avg)
    aspect_ids.extend(batch_ids)
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

# years = ['acl_2017', 'iclr_2017', 'iclr_2018', 'iclr_2019', 'iclr_2020', 'nips_2013', 'nips_2014', 'nips_2015', 'nips_2016', 'nips_2017', 'nips_2018']
years = ['acl_2017']
path_new = 'paper/recommend/%s.json'
apro_proc = Recommend_Processor()
for year in years:
    t_dataset = load_recommend('scores/RECOMMENDATION/' + year + '.json')
    test_dataset = Recommend_Dataset(t_dataset, word2id, fa1_bins, fa2_bins, fs1_bins, fs2_bins, fn1_bins, fn2_bins, fr1_bins, fr2_bins, fi1_bins, parameters['max_len'])



    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
        collate_fn=apro_proc.process
    )
    with open(path_new % year, 'w') as f:

        for batch_idx, (batch_txts, batch_lens, batch_fa1s, batch_fa2s, batch_fs1s, batch_fs2s, batch_fn1s, batch_fn2s, batch_fr1s, batch_fr2s, batch_fi1s, batch_tgt, batch_ids, batch_avg) in enumerate(test_loader):

            batch_txts =batch_txts.to(device)

            batch_fa1s = batch_fa1s.to(device)
            batch_fa2s = batch_fa2s.to(device)
            batch_fs1s = batch_fs1s.to(device)
            batch_fs2s = batch_fs2s.to(device)
            batch_fn1s = batch_fn1s.to(device)
            batch_fn2s = batch_fn2s.to(device)
            batch_fr1s = batch_fr1s.to(device)
            batch_fr2s = batch_fr2s.to(device)
            batch_fi1s = batch_fi1s.to(device)
            batch_tgt = batch_tgt.to(device)
            prob = model(batch_txts, batch_lens, batch_fa1s, batch_fa2s, batch_fs1s, batch_fs2s, batch_fn1s, batch_fn2s, batch_fr1s, batch_fr2s, batch_fi1s)
            symbols = prob.topk(1)[1].squeeze(1).tolist()
            for symbol, id_ in zip(symbols, batch_ids):
                tmp = {'id':id_, 'score': symbol + 1}
                f.write(json.dumps(tmp) + '\n')