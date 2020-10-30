import json
from tqdm import tqdm


# acl_aspects = ['RECOMMENDATION', 'APPROPRIATENESS','MEANINGFUL_COMPARISON','SOUNDNESS_CORRECTNESS','ORIGINALITY', 'IMPACT','CLARITY']




def load_recommend(path, train=False):
    words = []
    abstracts = []
    fa1 = []
    fa2 = []
    fs1 = []
    fs2 = []
    fn1 = []
    fn2 = []
    fr1 = []
    fr2 = []
    fi1 = []
    scores = []
    scores_avg = []
    ids = []
    with open(path, 'r') as f:
        for line in tqdm(f):
            tmp = json.loads(line)
            ids.append(tmp['id'])
            data = tmp["feature"]
            features = data['APPROPRIATENESS']
            fa1.append(features[0])
            fa2.append(features[1])
            features = data['MEANINGFUL_COMPARISON']
            fr1.append(features[0])
            fr2.append(features[1])
            features = data['SOUNDNESS_CORRECTNESS']
            fs1.append(features[0])
            fs2.append(features[1])
            features = data['ORIGINALITY']
            fn1.append(features[0])
            fn2.append(features[1])
            features = data['IMPACT']
            fi1.append(features[0])
            abstracts.append(data['CLARITY'])
            words.extend(features[2])
            scores.append(tmp['score']-1)
            scores_avg.append(tmp['score_avg'])
    if train:
        return words, [ids, fa1, fa2, fs1, fs2, fn1, fn2, fr1, fr2, fi1, abstracts, scores, scores_avg]
    else:
        return [ids, fa1, fa2, fs1, fs2, fn1, fn2, fr1, fr2, fi1, abstracts, scores, scores_avg]

def load_appro_sound_novelty(path, train=False):
    words = []
    abstracts = []
    f1 = []
    f2 = []
    scores = []
    scores_avg = []
    ids = []
    with open(path, 'r') as f:
        for line in tqdm(f):
            data = json.loads(line)
            ids.append(data['id'])
            features = data['feature']
            f1.append(features[0])
            f2.append(features[1])
            abstracts.append(features[2])
            words.extend(features[2])
            scores.append(data['score']-1)
            scores_avg.append(data['score_avg'])
            
    if train:
        return words, [ids, f1, f2, abstracts, scores, scores_avg]
    else:
        return [ids, f1, f2, abstracts, scores, scores_avg]


def load_related(path, train=False):
    words = []
    sents = []
    f1 = []
    f2 = []
    scores = []
    scores_avg = []
    ids = []
    with open(path, 'r') as f:
        for line in tqdm(f):
            data = json.loads(line)
            ids.append(data['id'])
            features = data['feature']
            f1.append(features[0])
            f2.append(features[1])
            # f3.append(features[2])
            sents.append(features[2])
            words.extend(features[2])
            scores.append(data['score']-1)
            scores_avg.append(data['score_avg'])
    if train:
        return words, [ids, f1, f2, sents, scores, scores_avg]
    else:
        return [ids, f1, f2, sents, scores, scores_avg]


def load_clarity(path, train=False):
    words = []
    sents = []
    scores = []
    scores_avg = []
    ids = []
    with open(path, 'r') as f:
        for line in tqdm(f):
            data = json.loads(line)
            ids.append(data['id'])
            features = data['feature']
            sents.append(features)
            words.extend(features)
            scores.append(data['score']-1)
            scores_avg.append(data['score_avg'])
    if train:
        return words, [ids, sents, scores, scores_avg]
    else:
        return [ids, sents, scores, scores_avg]

def load_impact(path, train=False):
    words = []
    abstracts = []
    f1 = []
    scores = []
    scores_avg = []
    ids = []
    with open(path, 'r') as f:
        for line in tqdm(f):
            data = json.loads(line)
            ids.append(data['id'])
            features = data['feature']
            f1.append(features[0])
            # f2.append(features[1])
            abstracts.append(features[2])
            words.extend(features[2])
            scores.append(data['score']-1)
            scores_avg.append(data['score_avg'])
    if train:
        return words, [ids, f1, abstracts, scores, scores_avg]
    else:
        return [ids, f1, abstracts, scores, scores_avg]
