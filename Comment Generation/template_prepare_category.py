import json
from tqdm import tqdm
from collections import defaultdict
from nltk import word_tokenize, sent_tokenize
years = ['acl_2017'] #, 'iclr_2017', 'iclr_2018', 'iclr_2019', 'iclr_2020', 'nips_2013', 'nips_2014', 'nips_2015', 'nips_2016', 'nips_2017', 'nips_2018']

path = '../dataset/Raw_data/Paper-review Corpus/%s/metadata.txt'

path_new = 'template_paper/%s.txt'
categories = ['SUMMARY', 'MEANINGFUL_COMPARISON', 'ORIGINALITY']
a2id = {}
id2a = {}
for i, a in enumerate(categories):
    a2id[a] = i
    id2a[i] = a
for year in years:
    print(year)
    filename = path % year
    wf =  open(path_new % year, 'w')
    with open(filename, 'r') as f:
        for line in tqdm(f):
            data = json.loads(line)
            id_ = data['id']
            title = data['paper_content']['title']
            tmp = {
                'id': id_,
                'title': title, 
                'reviews': {}
            }

            for i, review in enumerate(data["review"]):
                sents = []
                review_content = review["review"].replace("\n", " ").replace("\r", " ").strip().lower()
                tmp['reviews'][str(i)] = defaultdict(list)
                for review_sent in sent_tokenize(review_content):
                    w_token = word_tokenize(review_sent)
                    sents.append(w_token)
                for c in categories:
                    tmp['reviews'][str(i)][c] = sents
            wf.write(json.dumps(tmp) + '\n')
wf.close()