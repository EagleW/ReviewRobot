import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Appro_Sound(nn.Module):
    def __init__(self, embed_layer_word, embed_layer_f1, embed_layer_f2, input_dropout, layer_dropout, w_dim, f_dim, h_dim, num_class, **kwargs):
        super(Appro_Sound, self).__init__()
        self.embed_layer_word = embed_layer_word
        self.embed_layer_f1 = embed_layer_f1
        self.embed_layer_f2 = embed_layer_f2
        self.dropout = nn.Dropout(input_dropout)
        self.gru = nn.GRU(w_dim, h_dim//2, bidirectional=True)
        self.Ws = nn.Linear(h_dim, h_dim)
        self.v = nn.Linear(h_dim, 1)
        self.fc = nn.Linear(h_dim + f_dim * 2, num_class)
    
    def forward(self, batch_txts, batch_lens, batch_f1s, batch_f2s):
        sents = self.embed_layer_word(batch_txts)
        f1s = self.embed_layer_f1(batch_f1s)
        f2s = self.embed_layer_f2(batch_f2s)
        sents = self.dropout(sents)

        output = pack_padded_sequence(sents, batch_lens, batch_first=True)
        output, h = self.gru(output)
        output, _ = pad_packed_sequence(output, batch_first=True)
        batch_size, max_enc_len, _ = output.size()
        hidden = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2).transpose(0, 1)
        dec_proj = self.Ws(hidden).expand_as(output).contiguous()
        tmp = torch.tanh(output + dec_proj)
        attn_scores = self.v(tmp.view(batch_size * max_enc_len, -1))
        attn_scores = attn_scores.view(batch_size, max_enc_len)
        attn_scores = F.softmax(attn_scores, dim=1)
        context = attn_scores.unsqueeze(1).bmm(output).squeeze(1)
        prob = self.fc(torch.cat([context, f1s, f2s], 1))
        return prob


class Related(nn.Module):
    def __init__(self, embed_layer_word, embed_layer_f1, embed_layer_f2, input_dropout, layer_dropout, w_dim, f_dim, h_dim, num_class, **kwargs):
        super(Related, self).__init__()
        self.embed_layer_word = embed_layer_word
        self.embed_layer_f1 = embed_layer_f1
        self.embed_layer_f2 = embed_layer_f2
        self.gru = nn.GRU(w_dim, h_dim//2, bidirectional=True)
        self.Ws = nn.Linear(h_dim, h_dim)
        self.v = nn.Linear(h_dim, 1)
        self.fc = nn.Linear(h_dim + f_dim * 2, num_class)
    
    def forward(self, batch_txts, batch_lens, batch_f1s, batch_f2s):
        sents = self.embed_layer_word(batch_txts)
        f1s = self.embed_layer_f1(batch_f1s)
        f2s = self.embed_layer_f2(batch_f2s)

        output = pack_padded_sequence(sents, batch_lens, batch_first=True)
        output, h = self.gru(output)
        output, _ = pad_packed_sequence(output, batch_first=True)
        batch_size, max_enc_len, _ = output.size()
        hidden = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2).transpose(0, 1)
        dec_proj = self.Ws(hidden).expand_as(output).contiguous()
        tmp = torch.tanh(output + dec_proj)
        attn_scores = self.v(tmp.view(batch_size * max_enc_len, -1))
        attn_scores = attn_scores.view(batch_size, max_enc_len)
        attn_scores = F.softmax(attn_scores, dim=1)
        context = attn_scores.unsqueeze(1).bmm(output).squeeze(1)
        prob = self.fc(torch.cat([context, f1s, f2s], 1))
        return prob
    


class Clarity(nn.Module):
    def __init__(self, embed_layer_word, input_dropout, layer_dropout, w_dim, f_dim, h_dim, num_class, **kwargs):
        super(Clarity, self).__init__()
        self.embed_layer_word = embed_layer_word
        self.gru = nn.GRU(w_dim, h_dim//2, bidirectional=True)
        self.Ws = nn.Linear(h_dim, h_dim)
        self.v = nn.Linear(h_dim, 1)
        self.fc = nn.Linear(h_dim, num_class)
    
    def forward(self, batch_txts, batch_lens):
        sents = self.embed_layer_word(batch_txts)

        output = pack_padded_sequence(sents, batch_lens, batch_first=True)
        output, h = self.gru(output)
        output, _ = pad_packed_sequence(output, batch_first=True)
        batch_size, max_enc_len, _ = output.size()
        hidden = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2).transpose(0, 1)
        dec_proj = self.Ws(hidden).expand_as(output).contiguous()
        tmp = torch.tanh(output + dec_proj)
        attn_scores = self.v(tmp.view(batch_size * max_enc_len, -1))
        attn_scores = attn_scores.view(batch_size, max_enc_len)
        attn_scores = F.softmax(attn_scores, dim=1)
        context = attn_scores.unsqueeze(1).bmm(output).squeeze(1)
        prob = self.fc(context)
        return prob


class Impact(nn.Module):
    def __init__(self, embed_layer_word, embed_layer_f1, input_dropout, layer_dropout, w_dim, f_dim, h_dim, num_class, **kwargs):
        super(Impact, self).__init__()
        self.embed_layer_word = embed_layer_word
        self.embed_layer_f1 = embed_layer_f1
        self.gru = nn.GRU(w_dim, h_dim//2, bidirectional=True)
        self.Ws = nn.Linear(h_dim, h_dim)
        self.v = nn.Linear(h_dim, 1)
        self.fc = nn.Linear(h_dim + f_dim, num_class)
    
    def forward(self, batch_txts, batch_lens, batch_f1s):
        sents = self.embed_layer_word(batch_txts)
        f1s = self.embed_layer_f1(batch_f1s)

        output = pack_padded_sequence(sents, batch_lens, batch_first=True)
        output, h = self.gru(output)
        output, _ = pad_packed_sequence(output, batch_first=True)
        batch_size, max_enc_len, _ = output.size()
        hidden = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2).transpose(0, 1)
        dec_proj = self.Ws(hidden).expand_as(output).contiguous()
        tmp = torch.tanh(output + dec_proj)
        attn_scores = self.v(tmp.view(batch_size * max_enc_len, -1))
        attn_scores = attn_scores.view(batch_size, max_enc_len)
        attn_scores = F.softmax(attn_scores, dim=1)
        context = attn_scores.unsqueeze(1).bmm(output).squeeze(1)
        prob = self.fc(torch.cat([context, f1s], 1))
        return prob
    

class Recommend(nn.Module):
    def __init__(self, embed_layer_word, embed_layer_fa1, embed_layer_fa2, embed_layer_fs1, embed_layer_fs2, embed_layer_fn1, embed_layer_fn2, embed_layer_fr1, embed_layer_fr2, embed_layer_fi1, input_dropout, layer_dropout, w_dim, f_dim, h_dim, num_class, **kwargs):
        super(Recommend, self).__init__()
        self.embed_layer_word = embed_layer_word
        self.embed_layer_fa1 = embed_layer_fa1
        self.embed_layer_fa2 = embed_layer_fa2
        self.embed_layer_fs1 = embed_layer_fs1
        self.embed_layer_fs2 = embed_layer_fs2
        self.embed_layer_fn1 = embed_layer_fn1
        self.embed_layer_fn2 = embed_layer_fn2
        self.embed_layer_fr1 = embed_layer_fr1
        self.embed_layer_fr2 = embed_layer_fr2
        self.embed_layer_fi1 = embed_layer_fi1
        self.gru = nn.GRU(w_dim, h_dim//2, bidirectional=True)
        self.Ws = nn.Linear(h_dim, h_dim)
        self.v = nn.Linear(h_dim, 1)
        self.fc = nn.Linear(h_dim + f_dim * 9, num_class)
    
    def forward(self, batch_txts, batch_lens, batch_fa1s, batch_fa2s, batch_fs1s, batch_fs2s, batch_fn1s, batch_fn2s, batch_fr1s, batch_fr2s, batch_fi1s):
        sents = self.embed_layer_word(batch_txts)
        fa1 = self.embed_layer_fa1(batch_fa1s)
        fa2 = self.embed_layer_fa2(batch_fa2s)
        fs1 = self.embed_layer_fs1(batch_fs1s)
        fs2 = self.embed_layer_fs2(batch_fs2s)
        fn1 = self.embed_layer_fn1(batch_fn1s)
        fn2 = self.embed_layer_fn2(batch_fn2s)
        fr1 = self.embed_layer_fr1(batch_fr1s)
        fr2 = self.embed_layer_fr2(batch_fr2s)
        fi1 = self.embed_layer_fi1(batch_fi1s)

        output = pack_padded_sequence(sents, batch_lens, batch_first=True)
        output, h = self.gru(output)
        output, _ = pad_packed_sequence(output, batch_first=True)
        batch_size, max_enc_len, _ = output.size()
        hidden = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2).transpose(0, 1)
        dec_proj = self.Ws(hidden).expand_as(output).contiguous()
        tmp = torch.tanh(output + dec_proj)
        attn_scores = self.v(tmp.view(batch_size * max_enc_len, -1))
        attn_scores = attn_scores.view(batch_size, max_enc_len)
        attn_scores = F.softmax(attn_scores, dim=1)
        context = attn_scores.unsqueeze(1).bmm(output).squeeze(1)
        prob = self.fc(torch.cat([context, fa1, fa2, fs1, fs2, fn1, fn2, fr1, fr2, fi1], 1))
        return prob


