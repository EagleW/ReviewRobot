import torch
from torch.utils.data import Dataset
from .utils import vectorize_bin, vectorize_word


class Appro_Sound_Novelty_Dataset(Dataset):
    def __init__(self, dataset, word2id, f1_bins, f2_bins, max_len):
        self.ids, f1, f2, abstracts, self.scores, self.scores_avg = dataset
        self.vector_txts, self.lens = vectorize_word(abstracts, word2id, word2id['<pad>'], max_len)
        self.vector_f1s = vectorize_bin(f1, f1_bins)
        self.vector_f2s = vectorize_bin(f2, f2_bins)
        self.data_size = len(f1)
    
    def __getitem__(self, i):
        return self.vector_txts[i], self.lens[i], self.vector_f1s[i], self.vector_f2s[i], self.scores[i], self.ids[i], self.scores_avg[i]

    def __len__(self):
        return self.data_size


class Appro_Sound_Novelty_Processor(object):
    def __init__(self):
        pass
    
    def process(self, batch: list):
        batch.sort(key=lambda x: x[1], reverse=True)
        batch_cols = tuple(zip(*batch))

        batch_txts, batch_lens, batch_f1s, batch_f2s, batch_tgt, batch_ids, batch_avg = batch_cols
        batch_txts = torch.LongTensor(batch_txts)
        batch_lens = torch.LongTensor(batch_lens)
        batch_f1s = torch.LongTensor(batch_f1s)
        batch_f2s = torch.LongTensor(batch_f2s)
        batch_tgt = torch.LongTensor(batch_tgt)
        return batch_txts, batch_lens, batch_f1s, batch_f2s, batch_tgt, batch_ids, batch_avg


class Related_Dataset(Dataset):
    def __init__(self, dataset, word2id, f1_bins, f2_bins, max_len):
        self.ids, f1, f2, abstracts, self.scores, self.scores_avg = dataset
        self.vector_txts, self.lens = vectorize_word(abstracts, word2id, word2id['<pad>'], max_len)
        self.vector_f1s = vectorize_bin(f1, f1_bins)
        self.vector_f2s = vectorize_bin(f2, f2_bins)
        self.data_size = len(f1)
    
    def __getitem__(self, i):
        return self.vector_txts[i], self.lens[i], self.vector_f1s[i], self.vector_f2s[i], self.scores[i], self.ids[i], self.scores_avg[i]

    def __len__(self):
        return self.data_size


class Related_Processor(object):
    def __init__(self):
        pass
    
    def process(self, batch: list):
        batch.sort(key=lambda x: x[1], reverse=True)
        batch_cols = tuple(zip(*batch))

        batch_txts, batch_lens, batch_f1s, batch_f2s, batch_tgt, batch_ids, batch_avg = batch_cols
        batch_txts = torch.LongTensor(batch_txts)
        batch_lens = torch.LongTensor(batch_lens)
        batch_f1s = torch.LongTensor(batch_f1s)
        batch_f2s = torch.LongTensor(batch_f2s)
        batch_tgt = torch.LongTensor(batch_tgt)
        return batch_txts, batch_lens, batch_f1s, batch_f2s, batch_tgt, batch_ids, batch_avg


class Clarity_Dataset(Dataset):
    def __init__(self, dataset, word2id, max_len):
        self.ids, abstracts, self.scores, self.scores_avg = dataset
        self.vector_txts, self.lens = vectorize_word(abstracts, word2id, word2id['<pad>'], max_len)
        self.data_size = len(self.ids)
    
    def __getitem__(self, i):
        return self.vector_txts[i], self.lens[i], self.scores[i], self.ids[i], self.scores_avg[i]

    def __len__(self):
        return self.data_size


class Clarity_Processor(object):
    def __init__(self):
        pass
    
    def process(self, batch: list):
        batch.sort(key=lambda x: x[1], reverse=True)
        batch_cols = tuple(zip(*batch))

        batch_txts, batch_lens, batch_tgt, batch_ids, batch_avg = batch_cols
        batch_txts = torch.LongTensor(batch_txts)
        batch_lens = torch.LongTensor(batch_lens)
        batch_tgt = torch.LongTensor(batch_tgt)
        return batch_txts, batch_lens, batch_tgt, batch_ids, batch_avg


class Baseline_Dataset(Dataset):
    def __init__(self, dataset, word2id, max_len):
        self.ids, abstracts, self.scores, self.scores_avg = dataset
        self.vector_txts, self.lens = vectorize_word(abstracts, word2id, word2id['<pad>'], max_len)
        self.data_size = len(self.ids)
    
    def __getitem__(self, i):
        return self.vector_txts[i], self.lens[i], self.scores[i], self.ids[i], self.scores_avg[i]

    def __len__(self):
        return self.data_size


class Baseline_Processor(object):
    def __init__(self):
        pass
    
    def process(self, batch: list):
        batch.sort(key=lambda x: x[1], reverse=True)
        batch_cols = tuple(zip(*batch))

        batch_txts, batch_lens, batch_tgt, batch_ids, batch_avg = batch_cols
        batch_txts = torch.LongTensor(batch_txts)
        batch_lens = torch.LongTensor(batch_lens)
        batch_tgt = torch.LongTensor(batch_tgt)
        return batch_txts, batch_lens, batch_tgt, batch_ids, batch_avg


class Impact_Dataset(Dataset):
    def __init__(self, dataset, word2id, f1_bins, max_len):
        self.ids, f1, abstracts, self.scores, self.scores_avg = dataset
        self.vector_txts, self.lens = vectorize_word(abstracts, word2id, word2id['<pad>'], max_len)
        self.vector_f1s = vectorize_bin(f1, f1_bins)
        self.data_size = len(f1)
    
    def __getitem__(self, i):
        return self.vector_txts[i], self.lens[i], self.vector_f1s[i], self.scores[i], self.ids[i], self.scores_avg[i]

    def __len__(self):
        return self.data_size


class Impact_Processor(object):
    def __init__(self):
        pass
    
    def process(self, batch: list):
        batch.sort(key=lambda x: x[1], reverse=True)
        batch_cols = tuple(zip(*batch))

        batch_txts, batch_lens, batch_f1s, batch_tgt, batch_ids, batch_avg = batch_cols
        batch_txts = torch.LongTensor(batch_txts)
        batch_lens = torch.LongTensor(batch_lens)
        batch_f1s = torch.LongTensor(batch_f1s)
        batch_tgt = torch.LongTensor(batch_tgt)
        return batch_txts, batch_lens, batch_f1s, batch_tgt, batch_ids, batch_avg

class Recommend_Dataset(Dataset):
    def __init__(self, dataset, word2id, fa1_bins, fa2_bins, fs1_bins, fs2_bins, fn1_bins, fn2_bins, fr1_bins, fr2_bins, fi1_bins, max_len):
        self.ids, fa1, fa2, fs1, fs2, fn1, fn2, fr1, fr2, fi1, abstracts, self.scores, self.scores_avg = dataset
        self.vector_txts, self.lens = vectorize_word(abstracts, word2id, word2id['<pad>'], max_len)
        self.vector_fa1s = vectorize_bin(fa1, fa1_bins)
        self.vector_fa2s = vectorize_bin(fa2, fa2_bins)
        self.vector_fs1s = vectorize_bin(fs1, fs1_bins)
        self.vector_fs2s = vectorize_bin(fs2, fs2_bins)
        self.vector_fn1s = vectorize_bin(fn1, fn1_bins)
        self.vector_fn2s = vectorize_bin(fn2, fn2_bins)
        self.vector_fr1s = vectorize_bin(fr1, fr1_bins)
        self.vector_fr2s = vectorize_bin(fr2, fr2_bins)
        self.vector_fi1s = vectorize_bin(fi1, fi1_bins)
        self.data_size = len(fa1)
    
    def __getitem__(self, i):
        return self.vector_txts[i], self.lens[i], self.vector_fa1s[i], self.vector_fa2s[i], self.vector_fs1s[i], self.vector_fs2s[i], self.vector_fn1s[i], self.vector_fn2s[i], self.vector_fr1s[i], self.vector_fr2s[i], self.vector_fi1s[i], self.scores[i], self.ids[i], self.scores_avg[i]

    def __len__(self):
        return self.data_size


class Recommend_Processor(object):
    def __init__(self):
        pass
    
    def process(self, batch: list):
        batch.sort(key=lambda x: x[1], reverse=True)
        batch_cols = tuple(zip(*batch))

        batch_txts, batch_lens, batch_fa1s, batch_fa2s, batch_fs1s, batch_fs2s, batch_fn1s, batch_fn2s, batch_fr1s, batch_fr2s, batch_fi1s, batch_tgt, batch_ids, batch_avg = batch_cols
        batch_txts = torch.LongTensor(batch_txts)
        batch_lens = torch.LongTensor(batch_lens)

        batch_fa1s = torch.LongTensor(batch_fa1s)
        batch_fa2s = torch.LongTensor(batch_fa2s)
        batch_fs1s = torch.LongTensor(batch_fs1s)
        batch_fs2s = torch.LongTensor(batch_fs2s)
        batch_fn1s = torch.LongTensor(batch_fn1s)
        batch_fn2s = torch.LongTensor(batch_fn2s)
        batch_fr1s = torch.LongTensor(batch_fr1s)
        batch_fr2s = torch.LongTensor(batch_fr2s)
        batch_fi1s = torch.LongTensor(batch_fi1s)

        batch_tgt = torch.LongTensor(batch_tgt)
        return batch_txts, batch_lens, batch_fa1s, batch_fa2s, batch_fs1s, batch_fs2s, batch_fn1s, batch_fn2s, batch_fr1s, batch_fr2s, batch_fi1s, batch_tgt, batch_ids, batch_avg
