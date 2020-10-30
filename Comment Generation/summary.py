import json
import glob
from tqdm import tqdm
import statistics
import os
import math
from collections import defaultdict, Counter, OrderedDict
import pickle
from copy import deepcopy


def get_repre(pair):
    h, tmp = pair
    t = list(tmp)
    new_t = ''
    if len(t) == 1:
        new_t = t[0]
    elif len(t) == 2:
        new_t = t[0] + ' and ' + t[1]
    else:
        new_t = ', '.join(t[:-1]) + ' and ' + t[-1]
    return h, new_t

def get_repre_r(pair):
    h, t = pair
    t = list(t)
    new_t = ''
    if len(t) == 1:
        new_t = t[0]
    elif len(t) == 2:
        new_t = t[0] + ' and ' + t[1]
    else:
        new_t = ', '.join(t[:-1]) + ' and ' + t[-1]
    # print(new_t, h)
    return new_t, h
def get_whole(t):
    new_t = ''
    if len(t) == 1:
        new_t = t[0]
    elif len(t) == 2:
        new_t = t[0] + ' and ' + t[1]
    else:
        new_t = ', '.join(t[:-1]) + ' and ' + t[-1]
    return new_t

path = 'data/SUMMARY/test.json'
path_f = 'summary.txt'
wf = open(path_f, 'w')
count_ = 0
idds = []
with open(path, 'r') as f:
    for line in tqdm(f):
        output = []
        data = json.loads(line)
        diff_c, diff_rel, txts_list, c2t, c2num, r2num, c2idfs, r2idfs = data['src']
        score = data['score']
        if len(diff_c) == 0:
            continue 
        if len(diff_rel) == 0:
            continue 
        pid = data['pid']
        idds.append(pid)
        output.append('id: ' + pid + ' '+ data['title'] + '\n')
        output.append('score: ' + str(data['score']) + '\n')
        real_name = {}
        pair_used_for = defaultdict(set)
        compare = defaultdict(set)
        features = defaultdict(set)
        pair_e = defaultdict(set)
        output.append('Strengths:\n')
        flag = True
        for h, t, r in diff_rel:
            if r == "USED-FOR":
                try:
                    if c2t[t] in ["Method", "Material", "Metric"] and  c2t[h]  == 'Task':
                        pair_used_for[h].add(t)
                except:
                    pair_used_for[h].add(t)

            elif r == 'COMPARE':
                if h != t:
                    if h in compare:
                        compare[h].add(t)
                    else:
                        compare[t].add(h)
            elif r == 'FEATURE-OF':
                if h != t:
                    features[t].add(h)
            elif r == "EVALUATE-FOR":
                try:
                    if c2t[t] == "Method" and  c2t[h] in ["Material", "Metric"]:
                        pair_e[t].add(h)
                except:
                    pass
        sorted_relation = []
        r = "USED-FOR"
        for h, ts in pair_used_for.items():
            pair = (h, ts)
            score = 0
            for t in ts:
                rel = str((h, t, r))
                score += r2num[rel] / r2idfs.get(rel, 1) 
            sorted_relation.append((get_repre(pair), score))
        sorted_relation.sort(key=lambda x: x[1], reverse=True)
        for pair, count in sorted_relation:
            if score > 3:
                output.append('\tThis paper uses novel %s for %s . \n' % pair)
            else:
                output.append('\tThis paper uses %s for %s . \n' % pair)
            # output.append('\tTF_IDF:'+ str(count) + '\n\n')
            flag = False
        
        sorted_relation = []
        r = "COMPARE"
        for h, ts in compare.items():
            pair = (h, ts)
            score = 0
            for t in ts:
                rel = str((h, t, r))
                if rel not in r2num:
                    rel = str((t, h, r))
                score += r2num[rel] / r2idfs.get(rel, 1) 
            sorted_relation.append((get_repre(pair), score))
        sorted_relation.sort(key=lambda x: x[1], reverse=True)
        for pair, count in sorted_relation:
            output.append('\tThe paper compare %s with %s . \n' % pair)
            # output.append('\tTF_IDF:'+ str(count) + '\n\n')
            flag = False

        sorted_relation = []
        r = 'FEATURE-OF'
        for h, ts in features.items():
            pair = (h, ts)
            score = 0
            for t in ts:
                rel = str((t, h, r))
                score += r2num[rel] / r2idfs.get(rel, 1) 
            sorted_relation.append((get_repre_r(pair), score))
        sorted_relation.sort(key=lambda x: x[1], reverse=True)
        for pair, count in sorted_relation:
            output.append('\tThe paper uses %s for %s . \n' % pair)
            # output.append('\tTF_IDF:'+ str(count) + '\n\n')
            flag = False

        sorted_relation = []
        r = "EVALUATE-FOR"
        new_entities = []
        for h, ts in pair_e.items():
            pair = (h, ts)
            score = 0
            new_entities.append(h)
            for t in ts:
                rel = str((t, h, r))
                score += r2num[rel] / r2idfs.get(rel, 1) 
            sorted_relation.append((pair, score))
        sorted_relation.sort(key=lambda x: x[1], reverse=True)
        if flag and len(new_entities) > 0:
            output.append('\tThis paper proposes  a new %s. \n' % get_whole(new_entities))
        for pair, count in sorted_relation:
            output.append('\tThe authors then evaluate %s using %s. \n' % get_repre(pair))
            # output.append('\tTF_IDF:'+ str(count) + '\n\n')
            flag = False
        if len(output) == 0:
            continue
        if flag:
            metric = []
            method = []
            task = []
            material = []
            other = []
            for e in diff_c:
                if e not in c2t:
                    other.append(e)
                elif c2t[e] == "Method":
                    method.append(e)
                elif c2t[e] ==  "Material":
                    material.append(e)
                elif c2t[e] == "Metric":
                    metric.append(e)
                elif c2t[e]  == 'Task':
                    task.append(e)
                else:
                    other.append(e)
            method = sorted(method, key=lambda i: c2num[i]/c2idfs.get(i, 1), reverse=True)
            task = sorted(task, key=lambda i: c2num[i]/c2idfs.get(i, 1), reverse=True)
            material  = sorted(material, key=lambda i: c2num[i]/c2idfs.get(i, 1), reverse=True)
            o = sorted(other, key=lambda i: c2num[i]/c2idfs.get(i, 1), reverse=True)
            if len(method[:5]) > 0:
                output.append('\tThe paper proposes novel %s' % get_whole(method[:5] ))

                if len(task) > 0:
                    output.append('for %s.\n' % get_whole(task[:5]))
                else:
                    output.append('.\n')
                # output.append('\tTF_IDF:')
                # for m in method[:5]:
                #     output.append(str(c2num[m]/c2idfs.get(m, 1))+ ' ')
                # for m in task[:5]:
                #     output.append(str(c2num[m]/c2idfs.get(m, 1))+ ' ')
                # output.append('\n\n')
                flag = False
            elif len(other) > 0:
                output.append('\tThe paper proposes novel %s' % get_whole(other[:5]))

                if len(task) > 0:
                    output.append('for %s.\n' % get_whole(task[:5]))
                else:
                    output.append('.\n')
                # output.append('\tTF_IDF:')
                # for m in other[:5]:
                #     output.append(str(c2num[m]/c2idfs.get(m, 1))+ ' ')
                # for m in task[:5]:
                #     output.append(str(c2num[m]/c2idfs.get(m, 1))+ ' ')
                # output.append('\n\n')
                flag = False
            else:
                flag = True
            if flag:
                if len(task) > 0:
                    output.append('\tThe paper proposes novel %s .\n' % get_whole(task[:5]))
                    # output.append('\tTF_IDF:')
                    # for m in task[:5]:
                    #     output.append(str(c2num[m]/c2idfs.get(m, 1))+ ' ')
                    # output.append('\n\n')
            if len(material) > 0:
                output.append('\tThe authors uses %s .\n' % get_whole(material[:5]))
                # output.append('\tTF_IDF:')
                # for m in material[:5]:
                #     output.append(str(c2num[m]/c2idfs.get(m, 1))+ ' ')
                # output.append('\n\n')
            if len(metric) > 0:
                output.append('\tThe paper uses %s for evaluation.\n' % get_whole(metric))
                # output.append('\tTF_IDF:')
                # for m in metric[:5]:
                #     output.append(str(c2num[m]/c2idfs.get(m, 1))+ ' ')
                # output.append('\n\n')
        
        output.append('\n')
        output.append('Reference:\n')
        for txt in data['tgt']:
            output.append(txt + '\n')
        output.append('\n\n\n')
        wf.writelines(output)
        count_ += 1
        if count_ %5 == 0:
            wf.write('-'*100)
            wf.write('\n\n')
            print(idds)
            idds = []   
wf.close()