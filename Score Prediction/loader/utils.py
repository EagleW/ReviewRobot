from bisect import bisect
from collections import Counter

def create_recommend(words, min_freq):
    word2id, id2word = prepare_mapping(words, min_freq)
    mappings = {
        'fa1': [20,30,40,50,60,70],
        'fa2': [30,40,50,60,70,80,90,100],
        'fr1': [10,20],
        'fr2': [5,10,15,20,25,30,35],
        'fs1': [10,20,30,40,50],
        'fs2': [10,20,30,40,50],
        'fn1': [5,10,15,20,25,30,35],
        'fn2': [5,10,15,20,25,30,35,40],
        'fi1': [2,4,6,8],
        'word2id': word2id,
        'id2word': id2word
    }
    return mappings

def create_appropriate(words, min_freq):
    word2id, id2word = prepare_mapping(words, min_freq)
    mappings = {
        'f1': [20,30,40,50,60,70],
        'f2': [30,40,50,60,70,80,90,100],
        'word2id': word2id,
        'id2word': id2word
    }
    return mappings

def create_related(words, min_freq):
    word2id, id2word = prepare_mapping(words, min_freq)
    mappings = {
        'f1': list(range(10)),
        'f2': list(range(10)),
        # 'f1': [10,20],
        # 'f2': [5,10,15,20,25,30,35],
        'word2id': word2id,
        'id2word': id2word
    }
    return mappings

def create_sound(words, min_freq):
    word2id, id2word = prepare_mapping(words, min_freq)
    mappings = {
        'f1': [10,20,30],
        'f2': [10,20,30],
        'word2id': word2id,
        'id2word': id2word
    }
    return mappings

def create_novelty(words, min_freq):
    word2id, id2word = prepare_mapping(words, min_freq)
    mappings = {
        # 'f1': [5,10,15,20,25,30,35],
        # 'f2': [5,10,15,20,25,30,35,40],
        'f1': list(range(10)),
        'f2': list(range(10)),
        'word2id': word2id,
        'id2word': id2word
    }
    return mappings

def create_clarity(words, min_freq):
    word2id, id2word = prepare_mapping(words, min_freq)
    mappings = {
        'word2id': word2id,
        'id2word': id2word
    }
    return mappings

def create_baseline(words, min_freq):
    word2id, id2word = prepare_mapping(words, min_freq)
    mappings = {
        'word2id': word2id,
        'id2word': id2word
    }
    return mappings

def create_impact(words, min_freq):
    word2id, id2word = prepare_mapping(words, min_freq)
    mappings = {
        'f1': [2,4,6,8],
        'word2id': word2id,
        'id2word': id2word
    }
    return mappings


def create_mapping(freq, min_freq=0, max_vocab=50000):
    freq = freq.most_common(max_vocab)
    item2id = {
        '<pad>': 0,
        '<unk>': 1
        # '<sos>': 2,
        # '<eos>': 3
    }
    offset = len(item2id)
    for i, v in enumerate(freq):
        if v[1] > min_freq:
            item2id[v[0]] = i + offset
    id2item = {i: v for v, i in item2id.items()}
    return item2id, id2item


def create_dict(item_list):
    assert type(item_list) is list
    freq = Counter(item_list)
    return freq


def prepare_mapping(words, min_freq):
    words_freq = create_dict(words)
    word2id, id2word = create_mapping(words_freq, min_freq)
    print("Found %i unique words (%i in total)" % (
        len(word2id), sum(len(x) for x in words)
    ))

    return word2id, id2word

def word2idx(word2id, w):
  return word2id[w] if w in word2id else word2id['<unk>']


def vectorize_word(sents, word2id, pad_id, max_len):
    vectors = []
    lens = []
    for sent in sents:
        tmp = [word2idx(word2id, w) for w in sent if not w.isnumeric()]
        vector = tmp[:max_len] + [0] * (max_len - len(tmp))
        vectors.append(vector)
        lens.append(len(tmp[:max_len]))
    return vectors, lens

def vectorize_bin(fs, f_bins):
    vectors = []
    for f in fs:
        vectors.append(bisect(f_bins, f))
    return vectors