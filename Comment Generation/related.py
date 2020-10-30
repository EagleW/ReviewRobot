import json
import glob
from tqdm import tqdm
import statistics
import os
import math
from collections import defaultdict, Counter
import pickle
from copy import deepcopy

path = 'data/MEANINGFUL_COMPARISON/test.json'

path_f = 'Related_work.txt'
wf = open(path_f, 'w')
count_ = 0
idds = []
with open(path, 'r') as f:
    for line in f:
        used_e = []
        data = json.loads(line)
        src_txt, used, tmp2id, same_tmp2id, id2paper_t, c2num, r2num  = data['src']
        if 'cluster' not in tmp2id:
            continue
        pid = data['pid']
        idds.append(pid)
        wf.write('id: ' + pid + ' '+ data['title'] + '\n')
        wf.write('Result:\n')
        wf.write('The following related papers are missing:\n')
        count = 1
        sorted_cluster = []
        for entity in tmp2id['cluster']:
            score = c2num[entity] / len(tmp2id['cluster'][entity])
            sorted_cluster.append((entity, score))
        sorted_cluster.sort(key=lambda x: x[1], reverse=True)
        flag = True
        for entity, s in sorted_cluster:
            titles = []
            for id_ in sorted(tmp2id['cluster'][entity], reverse=True):
                if id_ in used:
                    titles.append(id2paper_t[id_])
            if len(titles) > 0 and entity not in used_e:
                if flag or s >=0.5:
                    used_e.append(entity)
                    wf.write('(%d) About %s\n'% (count, entity))
                    tmp = []
                    cc = 0
                    for t in titles:
                        if t.lower() not in tmp and cc < 5:
                            wf.write('\t' + t + '\n')
                            cc += 1
                            tmp.append(t.lower())
                    count += 1
                    wf.write('\n')
                    flag = False
        if 'relation' in tmp2id:
            sorted_relation = []
            for rel in tmp2id['relation']:
                score = r2num[rel] / len(tmp2id['relation'][rel])
                sorted_relation.append((rel, score))
            sorted_relation.sort(key=lambda x: x[1], reverse=True)
            for rel, s in sorted_relation:
                titles = []
                for id_ in sorted(tmp2id['relation'][rel], reverse=True):
                    if id_ in used:
                        titles.append(id2paper_t[id_])
                if len(titles) > 0 and rel not in used_e:
                    if s >=0.5:
                        used_e.append(rel)
                        wf.write('(%d) About %s\n'% (count, rel))
                        tmp = []
                        cc = 0
                        for t in titles:
                            if t not in t or cc < 5:
                                wf.write('\t' + t + '\n')
                                cc += 1
                        count += 1
                        wf.write('\n')
        if 'cluster' in same_tmp2id:
            sorted_cluster = []
            for entity in same_tmp2id['cluster']:
                score = c2num[entity] / len(same_tmp2id['cluster'][entity])
                sorted_cluster.append((entity, score))
            sorted_cluster.sort(key=lambda x: x[1], reverse=True)
            flag = True
            for entity, s in sorted_cluster:
                titles = []
                for id_ in sorted(same_tmp2id['cluster'][entity], reverse=True):
                    if id_ in used:
                        titles.append(id2paper_t[id_])
                if len(titles) > 0 and entity not in used_e:
                    if s >=0.5:
                        used_e.append(entity)
                        wf.write('(%d) About %s\n'% (count, entity))
                        tmp = []
                        cc = 0
                        for t in titles:
                            if t.lower() not in tmp and cc < 5:
                                wf.write('\t' + t + '\n')
                                cc += 1
                                tmp.append(t.lower())
                        count += 1
                        wf.write('\n')
        if 'relation' in same_tmp2id:
            sorted_relation = []
            for rel in same_tmp2id['relation']:
                score = r2num[rel] / len(same_tmp2id['relation'][rel])
                sorted_relation.append((rel, score))
            sorted_relation.sort(key=lambda x: x[1], reverse=True)
            for rel, s in sorted_relation:
                titles = []
                for id_ in sorted(same_tmp2id['relation'][rel], reverse=True):
                    if id_ in used:
                        titles.append(id2paper_t[id_])
                if len(titles) > 0 and rel not in used:
                    if s >=0.5:
                        used_e.append(rel)
                        wf.write('(%d) About %s\n'% (count, rel))
                        tmp = []
                        cc = 0
                        for t in titles:
                            if t.lower() not in tmp and cc < 5:
                                wf.write('\t' + t + '\n')
                                cc += 1
                                tmp.append(t.lower())
                        count += 1
                        wf.write('\n')
        wf.write('\n')
        wf.write('Reference:\n')
        for txt in data['tgt']:
            wf.write(txt + '\n')
        wf.write('\n\n\n')
        count_ += 1
        if count_ %5 == 0:
            wf.write('-'*100)
            wf.write('\n\n')
            print(idds)
            idds = []       
wf.close()
