import json
from tqdm import tqdm
from collections import defaultdict
from random import shuffle
from statistics import mean 


path_kb = 'template_comment/%s.json'
path_review = 'template_paper/%s.txt'
path_score = '../Score Prediction/paper/%s/%s.json'
score_name = ['recommend', 'novelty', 'related']
aspects = ['SUMMARY', 'ORIGINALITY','MEANINGFUL_COMPARISON'] 
s2a = {}
a2id = {}
for i, (s, a) in enumerate(zip(score_name, aspects)):
    a2id[a] = i
    s2a[s] = a
def filter_t(t):
    t = t.replace('.', '_')
    t = t.replace(' ', '_')
    t = t.replace('(', '_lfp_')
    t = t.replace(')', '_rfp_')
    return t

def combine_entity(kb, tgt):
    tmp_tgt = []
    for sent in tgt:
        tmp_tgt.extend(sent)
    txts = ' '.join(tmp_tgt)
    new_entities = set()
    new_entities_rels = set()
    tmp_entities_rels = set()
    # remains = set()
    new_rels = set()
    tmp_rels = set()
    new_txts = []
    entities = set(kb[0])
    rels = kb[1]
    src_txt = kb[2]
    e2c = kb[3]
    c2t = kb[4]
    new_src = []
    first = set()
    for mention in e2c:
        if e2c[mention] == '':
            e2c[mention] = mention
    for entity in entities:
        new_entity = filter_t(entity)
        if entity in txts:
            txts = txts.replace(entity, new_entity)
            new_entities.add(new_entity)
            first.add(entity)

    entity2rel = {}
    for rel in rels:
        new_rel0 = filter_t(e2c[rel[0]])
        new_rel1 = filter_t(e2c[rel[1]])
        entities.add(e2c[rel[1]])
        new_rel = (new_rel0, new_rel1, rel[2])
        entity2rel[e2c[rel[0]]] = (new_rel0, new_rel1, rel[2])
        entity2rel[e2c[rel[1]]] = (new_rel0, new_rel1, rel[2])
        if rel[0] in txts:
            txts = txts.replace(rel[0], new_rel0)
            new_entities_rels.add(new_rel0)
            new_entities_rels.add(new_rel1)
            first.add(new_rel0)
            first.add(new_rel1)
            new_rels.add(new_rel)
        else:
            tmp_entities_rels.add(new_rel0)
            tmp_entities_rels.add(new_rel1)
            tmp_rels.add(new_rel)

        if rel[1] in txts:
            txts = txts.replace(rel[1], new_rel1)
            new_entities_rels.add(new_rel0)
            new_entities_rels.add(new_rel1)
            first.add(new_rel0)
            first.add(new_rel1)
            new_rels.add(new_rel)
        else:
            tmp_entities_rels.add(new_rel0)
            tmp_entities_rels.add(new_rel1)
            tmp_rels.add(new_rel)
    
    mentions = []
    for mention in e2c:
        if mention in txts:
            entity = e2c[mention]
            new_entity = filter_t(entity)
            txts = txts.replace(mention, new_entity)
            if entity in entities:
                mentions.append(mention)
                new_entities.add(new_entity)
                first.add(new_entity)
            if entity in entity2rel:
                new_rel = entity2rel[entity]
                if new_rel in tmp_rels:
                    tmp_rels.remove(new_rel)
                    new_rels.add(new_rel)
                    first.add(new_rel0)
                    first.add(new_rel1)
                    new_entities_rels.add(new_rel[0])
                    new_entities_rels.add(new_rel[1])
    count = 0
    for sent in src_txt:
        flag = True
        for entity in entities:
            if entity in sent:
                flag = False
                break
        if flag:
            for mention in mentions:
                if mention in sent:
                    flag = False
                    break
            if flag:
                continue
        for vocab in sent:
            if ' ' in vocab:
                new_src.append(filter_t(e2c[vocab]))
            elif not vocab.isdigit() and len(vocab) > 0:
                new_src.append(vocab)
        count += len(sent)
    if '' in new_entities:
        new_entities.remove('')
    if '' in new_entities_rels:
        new_entities_rels.remove('')
    new_txts = []
    count = 0
    for sent in tgt:
        second = set()
        txt = ' '.join(sent)
        for mention in e2c:
            if mention in txt:
                entity = e2c[mention]
                if entity in first:
                    new_entity = filter_t(entity)
                    txt = txt.replace(mention, new_entity)
                else:
                    second.add(mention)
        for mention in second:
            entity = e2c[mention]
            new_entity = filter_t(entity)
            txt = txt.replace(mention, new_entity)
        txt = txt.split()
        new_txts.extend(txt)
        count += len(txt)

    if len(new_rels) > 0:
        return new_entities, new_entities_rels, new_rels, new_src, new_txts, len(new_entities|new_entities_rels), c2t, True
    else:
        return new_entities, tmp_entities_rels, tmp_rels, new_src, new_txts, len(new_entities), c2t, False


years = ['acl_2017']#, 'iclr_2017', 'iclr_2018', 'iclr_2019', 'iclr_2020', 'nips_2013', 'nips_2014', 'nips_2015', 'nips_2016', 'nips_2017', 'nips_2018']
train_pairs = [[],[],[]]
valid_pairs = [[],[],[]]
test_pairs = [[],[],[]]
count = []
for year in years:
    print(year)
    dataset_pairs = [[],[],[]]
    comments = {}
    total = [0,0,0]
    pid2review = [[],[],[]]

    with open(path_review % year, 'r') as f:
        for line in tqdm(f):
            data = json.loads(line)
            id_ = data['id']
            data['scores'] = {}
            comments[id_] = data

    for s in score_name:
        with open(path_score % (s, year), 'r') as f:
            for line in tqdm(f):
                data = json.loads(line)
                id_ = data['id']
                if 'nips_2013' in year:
                    id_ = id_[4:]
                score = data['score']
                if id_ in comments:
                    comments[id_]['scores'][s2a[s]] = score
    
    with open(path_kb % year, 'r') as f:
        for line in tqdm(f):
            data = json.loads(line)
            id_ = data['id']
            if id_== '318':
                print(123)
            if 'nips' in year:
                id_ = id_[4:]
            if id_ not in comments:
                continue
            scores = comments[id_]['scores']
            comment = comments[id_]["reviews"]
            title = data['title']
            review_tmp = defaultdict(list) 
            for review_id in comment:
                for type_ in comment[review_id]:
                    # count += 1
                    review_tmp[type_].extend(comment[review_id][type_])
                    review_tmp[type_].append('<|>')
            for type_ in review_tmp:
                review_text = review_tmp[type_]
                kb = data[type_]
                tgt = []
                tmp = []
                for sent in review_text:
                    if sent == '<|>':
                        tgt.append(' '.join(tmp))
                        tmp = []
                    else:
                        tmp.extend(sent)
                if type_ == 'MEANINGFUL_COMPARISON':
                    if kb[0] is None:
                        continue
                flag = False
                score = scores[type_]
                tmp = {
                    'pid': year + '|' + id_,
                    'type': type_,
                    'tgt': tgt,
                    'src': kb,
                    'score': score,
                    'title': title
                }
                total[a2id[type_]] += 1
                dataset_pairs[a2id[type_]].append(json.dumps(tmp))
    if year == 'acl_2017' or year =='iclr_2017':
        for i in range(3):
            test_pairs[i].extend(dataset_pairs[i])
    else:
        for i in range(3):
            train_pairs[i].extend(dataset_pairs[i])

output = 'data/%s/'
test = []
for i in range(3):
    print(len(train_pairs[i]), len(valid_pairs[i]), len(test_pairs[i]))
    test.extend(test_pairs[i])
    with open((output % aspects[i]) + 'test.json', 'w') as f:
        for line in test_pairs[i]:
            f.write(line + '\n')
print()