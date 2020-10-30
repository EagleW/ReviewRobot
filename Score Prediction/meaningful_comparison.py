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
from model.RNN_torch import Related
from sklearn.metrics import accuracy_score, mean_squared_error
from loader.utils import create_related
from loader.data_loader import load_related
from loader.dataset import Related_Dataset, Related_Processor

parser = argparse.ArgumentParser()
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
    "--model_dp", default="models/related_models/",
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

parameters['data'] = 'acl_2017'
model_dir = args.model_dp
model_name = ['RNN']
model_dir = args.model_dp
for k, v in parameters.items():
    if v == "":
        continue
    if k == 'pre_emb':
        v = os.path.basename(v)
    model_name.append('='.join((k, str(v))))
model_dir = os.path.join(model_dir,  parameters['data'] +','.join(model_name[:-1]))
os.makedirs(model_dir, exist_ok=True)


parameters['num_class'] = 5

training_log_path = os.path.join(model_dir, 'training_log.txt')
if os.path.exists(training_log_path):
    os.remove(training_log_path)
f = open(training_log_path, 'w')
sys.stdout = Tee(sys.stdout, f)

if args.load:
    state = pickle.load(open(args.data_path + '/related_dataset.pth', 'rb'))
    mappings = state['mappings']
    r_dataset = state['r_dataset']
    v_dataset = state['v_dataset']
    t_dataset = state['t_dataset']
else:
    words, r_dataset = load_related(parameters['data'] + '/MEANINGFUL_COMPARISON_train.json', True)
    mappings = create_related(words, parameters['freq'])
    v_dataset = load_related(parameters['data'] + '/MEANINGFUL_COMPARISON_valid.json')
    t_dataset = load_related(parameters['data'] + '/MEANINGFUL_COMPARISON_test.json')
    state = {
        'mappings': mappings,
        'r_dataset': r_dataset,
        'v_dataset': v_dataset,
        't_dataset': t_dataset
    }
    pickle.dump(state, open(args.data_path + '/related_dataset.pth', "wb"))

word2id = mappings['word2id']
id2word = mappings['id2word']
vocab_size = len(mappings['id2word'])

f1_bins = mappings['f1']
f1_size = len(f1_bins) + 1

f2_bins = mappings['f2']
f2_size = len(f2_bins) + 1


device = torch.device("cuda:1" if torch.cuda.is_available() and parameters['gpu'] else "cpu")

train_dataset = Related_Dataset(r_dataset, word2id, f1_bins, f2_bins, parameters['max_len'])
valid_dataset = Related_Dataset(v_dataset, word2id, f1_bins, f2_bins, parameters['max_len'])
test_dataset = Related_Dataset(t_dataset, word2id, f1_bins, f2_bins, parameters['max_len'])

apro_proc = Related_Processor()

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=False,
    collate_fn=apro_proc.process
)

val_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=False,
    collate_fn=apro_proc.process
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=False,
    collate_fn=apro_proc.process
)

embed_layer_word = nn.Embedding(vocab_size, args.word_dim, padding_idx=0, sparse=False)
embed_layer_f1 = nn.Embedding(f1_size, args.feature_dim, padding_idx=0, sparse=False)
embed_layer_f2 = nn.Embedding(f2_size, args.feature_dim, padding_idx=0, sparse=False)

model = Related(embed_layer_word, embed_layer_f1, embed_layer_f2, **parameters)
criterion = nn.CrossEntropyLoss().to(device)
model = model.to(device)
optimizer = torch.optim.RMSprop(model.parameters(), args.lr_rate, weight_decay=0.9)

best_dev = np.inf
num_epochs = args.num_epochs
start_epoch = 0
epoch_examples_total = len(train_dataset)
print('train batches', epoch_examples_total)

for epoch in range(num_epochs):
    time_epoch_start = time.time()
    model.train(True)
    epoch_loss = 0
    for batch_idx, (batch_txts, batch_lens, batch_f1s, batch_f2s, batch_tgt, batch_ids, _) in enumerate(train_loader):
        batch_txts =batch_txts.to(device)
        batch_f1s = batch_f1s.to(device)
        batch_f2s= batch_f2s.to(device)
        batch_tgt = batch_tgt.to(device)
        prob = model(batch_txts, batch_lens, batch_f1s, batch_f2s)
        batch_loss = criterion(prob, batch_tgt)

        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        epoch_loss += batch_loss.item() * batch_txts.size(0)
        loss = batch_loss.item()
        optimizer.step()

        sys.stdout.write(
            '%d batches processed. current batch loss: %f\r' %
            (batch_idx, loss)
        )
        sys.stdout.flush()

    epoch_loss_avg = epoch_loss / float(epoch_examples_total)
    # print('Epoch: ', epoch, 'Train Loss: ', epoch_loss_avg)
    
    dev_loss = 0
    model.eval()
    for batch_idx, (batch_txts, batch_lens, batch_f1s, batch_f2s, batch_tgt, batch_ids, _) in enumerate(val_loader):
        batch_txts =batch_txts.to(device)
        batch_f1s = batch_f1s.to(device)
        batch_f2s= batch_f2s.to(device)
        batch_tgt = batch_tgt.to(device)
        prob = model(batch_txts, batch_lens, batch_f1s, batch_f2s)
        batch_loss = criterion(prob, batch_tgt)

        dev_loss += batch_loss.item() * batch_txts.size(0)
        loss = batch_loss.item()
        sys.stdout.flush()

    epoch_loss_avg = dev_loss / float(epoch_examples_total)
    # print('Epoch: ', epoch, 'Test Loss: ', epoch_loss_avg)
    if dev_loss < best_dev:
        best_dev = dev_loss
        state = {
            'epoch': start_epoch + epoch + 1,
            'model': model.state_dict(),
            'best_prec1': best_dev
        }
        torch.save(state, os.path.join(model_dir,'best_dev_model.pth.tar'))
state = torch.load(os.path.join(model_dir,'best_dev_model.pth.tar'))
state_dict = state['model']
model.load_state_dict(state_dict)

aspect_all_ys = []
aspect_all_ys_ = []
aspect_all_ys_avg = []
for batch_idx, (batch_txts, batch_lens, batch_f1s, batch_f2s, batch_tgt, batch_ids, batch_avg) in enumerate(test_loader):
    batch_txts =batch_txts.to(device)
    batch_f1s = batch_f1s.to(device)
    batch_f2s= batch_f2s.to(device)
    batch_tgt = batch_tgt.to(device)
    prob = model(batch_txts, batch_lens, batch_f1s, batch_f2s)
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