import json
import glob
from tqdm import tqdm
import statistics
import os
import math
from collections import defaultdict, Counter
import pickle
import matplotlib.pyplot as plt
from copy import deepcopy


def get_sentence(ids, sentence):
    ids = sorted(list(set(ids)))
    sents_list = []
    sents = []
    for id_ in sorted(list(ids)):
        sents.extend(sentence[id_])
        sents_list.append(sentence[id_])
    return sents, sents_list

def merge_cluster(source_tmp, target):
    source_cluster2entity = deepcopy(source_tmp['cluster2entity'])
    source_entity2cluster = deepcopy(source_tmp['entity2cluster'])
    cluster2entity = defaultdict(set)
    cluster2type = {}
    cluster2num = {}
    for c in target['cluster2entity']:
        cluster2entity[c] = set(target['cluster2entity'][c])
    entity2cluster = deepcopy(target['entity2cluster'])
    entity2cluster.update(source_tmp['entity2cluster'])

    for max_tag in target['cluster2entity']:
        if max_tag in source_entity2cluster:
            max_ntag = source_entity2cluster[max_tag]
        else:
            max_ntag = max_tag
        if max_ntag == '':
            continue
        update_sets = []
        for cur_tag in cluster2entity[max_tag]:
            if cur_tag in source_entity2cluster:
                if cur_tag == max_ntag:
                    update_sets.extend(list(source_cluster2entity[cur_tag]))
                elif cur_tag == source_entity2cluster[cur_tag]:
                    update_sets.extend(list(source_cluster2entity[cur_tag]))
                    source_cluster2entity.pop(cur_tag)
                else:
                    update_sets.append(cur_tag)
            else:
                update_sets.append(cur_tag)
        for cur_tag in update_sets:
            cluster2entity[max_ntag].add(cur_tag)
            entity2cluster[cur_tag] = max_ntag
            if cur_tag in source_entity2cluster:
                source_entity2cluster[cur_tag] = max_ntag
    
    for max_ntag in source_cluster2entity:
        if max_ntag == '':
            continue
        for cur_tag in source_cluster2entity[max_ntag]:
            cluster2entity[max_ntag].add(cur_tag)
            entity2cluster[cur_tag] = max_ntag
    related_e2c = deepcopy(target['entity2cluster'])
    idea_e2c = deepcopy(source_tmp['entity2cluster'])
    for max_ntag in cluster2entity:
        if max_ntag in source_tmp['entity2cluster']:
            if max_ntag == '':
                continue
            for cur_tag in cluster2entity[max_ntag]:
                if cur_tag in target['entity2cluster']:
                    related_e2c[cur_tag] = max_ntag
        else:
            update_target = set()
            max_s_tag = ''
            for cur_tag in cluster2entity[max_ntag]:
                if cur_tag in target['entity2cluster']:
                    update_target.add(cur_tag)
                if cur_tag in source_tmp['cluster2entity']:
                    if len(cur_tag) > len(max_s_tag):
                        max_s_tag = cur_tag
            assert cur_tag != ''
            if max_s_tag != '':
                for cur_tag in update_target:
                    related_e2c[cur_tag] = max_s_tag

        if max_ntag in target['entity2cluster']:
            if max_ntag == '':
                continue
            for cur_tag in cluster2entity[max_ntag]:
                if cur_tag in source_tmp['entity2cluster']:
                    idea_e2c[cur_tag] = max_ntag
        else:
            update_source = set()
            max_t_tag = ''
            for cur_tag in cluster2entity[max_ntag]:
                if cur_tag in source_tmp['entity2cluster']:
                    update_source.add(cur_tag)
                if cur_tag in target['cluster2entity']:
                    if len(cur_tag) > len(max_t_tag):
                        max_t_tag = cur_tag
            assert cur_tag != ''
            if max_t_tag != '':
                for cur_tag in update_source:
                    idea_e2c[cur_tag] = max_t_tag
    for c in cluster2entity:
        count = 0
        type_c = Counter()
        for e in cluster2entity[c]:
            if e in source_tmp['entity2num']:
                count += source_tmp['entity2num'][e]
            if e in target['entity2num']:
                count += target['entity2num'][e]
            if e in source_tmp['entity2type']:
                type_c[source_tmp['entity2type'][e]] += 1
            elif e in target['entity2type']:
                type_c[target['entity2type'][e]] += 1
        if count == 0:
            cluster2num[c] = 1
        else:
            cluster2num[c] = count
        if len(type_c) == 0:
            cluster2type[c] = "OtherScientificTerm"
        else:
            cluster2type[c] = type_c.most_common(1)[0][0]
    total_kb = {
        'cluster2entity' :cluster2entity, 
        'entity2cluster': entity2cluster,
        'related_e2c':related_e2c,
        'idea_e2c':idea_e2c,
        'cluster2type': cluster2type,
        'cluster2num': cluster2num
        }
    return total_kb

def diff_cluster(idea_kb, related_kb, t_e2c, t_c2t, c2n):
    diff_c = {}
    same_c = {}
    c2type = {}
    c2num = {}
    ids = []
    for max_tag in idea_kb['cluster2entity']:
        flag = True
        if max_tag in related_kb['entity']:
            same_c[max_tag] =  deepcopy(idea_kb['cluster2entity'][max_tag])
            if max_tag in t_e2c:
                max_tag1 = t_e2c[max_tag]
                c2type[max_tag] = t_c2t[max_tag1]
                c2num[max_tag] = c2n[max_tag1]
            else:
                type_c = Counter()
                count = 0
                for e in idea_kb[max_tag]:
                    if e in idea_kb['entity2type']:
                        type_c[idea_kb['entity2type'][e]] += 1
                    if e in idea_kb['entity2num']:
                        count += idea_kb['entity2num'][e]
                if count == 0:
                    c2num[max_tag] = 1
                else:
                    c2num[max_tag] = count
                if len(type_c) == 0:
                    c2type[max_tag] = "OtherScientificTerm"
                else:
                    c2type[max_tag] = type_c.most_common(1)[0][0]
            continue
        else:
            type_c = Counter()
            count = 0
            for cur_tag in idea_kb['cluster2entity'][max_tag]:
                if cur_tag in related_kb['cluster2entity'] and cur_tag != max_tag:
                    flag = False
                if cur_tag in idea_kb['entity2type']:
                    type_c[idea_kb['entity2type'][cur_tag]] += 1
                if cur_tag in idea_kb['entity2num']:
                    count += idea_kb['entity2num'][cur_tag]
            if count == 0:
                c2num[max_tag] = 1
            else:
                c2num[max_tag] = count
            if len(type_c) == 0:
                c2type[max_tag] = "OtherScientificTerm"
            else:
                c2type[max_tag] = type_c.most_common(1)[0][0]
        if flag:
            diff_c[max_tag] =  deepcopy(idea_kb['cluster2entity'][max_tag])
            ids.extend(idea_kb['cluster2sent'][max_tag])
        else:
            same_c[max_tag] =  deepcopy(idea_kb['cluster2entity'][max_tag])
    return diff_c, same_c, set(ids), c2type, c2num


def diff_cluster_b(idea_kb, back_kb, t_e2c, t_c2t, c2n):
    diff_c = {}
    c2type = {}
    c2num = {}
    ids = []
    e2c = {}
    new_tag = ''
    for max_tag in idea_kb['cluster2entity']:
        flag = True
        if max_tag in back_kb['entity']:
            new_tag = back_kb['entity2cluster'][max_tag]
            if max_tag in t_e2c:
                max_tag1 = t_e2c[max_tag]
                c2type[max_tag] = t_c2t[max_tag1]
                c2num[max_tag] = c2n[max_tag1]
                for e in idea_kb['cluster2entity'][max_tag]:
                    e2c[e] = new_tag
            else:
                type_c = Counter()
                count = 0
                for e in idea_kb['cluster2entity'][max_tag]:
                    if e in idea_kb['entity2type']:
                        type_c[idea_kb['entity2type'][e]] += 1
                    if e in idea_kb['entity2num']:
                        count += idea_kb['entity2num'][e]
                    e2c[e] = new_tag
                if count == 0:
                    c2num[max_tag] = 1
                else:
                    c2num[max_tag] = count
                if len(type_c) == 0:
                    c2type[max_tag] = "OtherScientificTerm"
                else:
                    c2type[max_tag] = type_c.most_common(1)[0][0]
            continue
        else:
            new_tag = ''
            type_c = Counter()
            count = 0
            for cur_tag in idea_kb['cluster2entity'][max_tag]:
                if cur_tag in back_kb['cluster2entity'] and cur_tag != max_tag:
                    flag = False
                    if len(new_tag) < len(cur_tag):
                        new_tag = cur_tag
                if cur_tag in idea_kb['entity2type']:
                    type_c[idea_kb['entity2type'][cur_tag]] += 1
                if cur_tag in idea_kb['entity2num']:
                    count += idea_kb['entity2num'][cur_tag]
            if new_tag == '':
                new_tag = max_tag
            for cur_tag in idea_kb['cluster2entity'][max_tag]:
                e2c[cur_tag] = new_tag
            if count == 0:
                c2num[max_tag] = 1
            else:
                c2num[max_tag] = count
            if len(type_c) == 0:
                c2type[max_tag] = "OtherScientificTerm"
            else:
                c2type[max_tag] = type_c.most_common(1)[0][0]
        if flag:
            diff_c[max_tag] =  deepcopy(idea_kb['cluster2entity'][max_tag])
            ids.extend(idea_kb['cluster2sent'][max_tag])
    return diff_c, set(ids), c2type, c2num, e2c


def diff_relation_b(idea_kb, back_kb, e2c):
    diff_rel = []
    ids = []
    r2num = {}
    for head in idea_kb['relations']:
        if e2c[head] in back_kb['relations']:
            for tail in idea_kb['relations'][head]:
                if e2c[tail] in back_kb['relations'][e2c[head]]:
                    for rel in idea_kb['relations'][head][tail]:
                        if rel not in back_kb['relations'][e2c[head]][e2c[tail]]:
                            diff_rel.append((head, tail, rel))
                            ids.extend(idea_kb['relation2sent'][head][tail][rel])
                            r2num[str((head, tail, rel))] = idea_kb['relation2num'][head][tail][rel]

                else:
                    for rel in idea_kb['relations'][head][tail]:
                        diff_rel.append((head, tail, rel))
                        ids.extend(idea_kb['relation2sent'][head][tail][rel])
                        r2num[str((head, tail, rel))] = idea_kb['relation2num'][head][tail][rel]
        else:
            for tail in idea_kb['relations'][head]:
                for rel in idea_kb['relations'][head][tail]:
                    diff_rel.append((head, tail, rel))
                    r2num[str((head, tail, rel))] = idea_kb['relation2num'][head][tail][rel]
                    ids.extend(idea_kb['relation2sent'][head][tail][rel])
    return diff_rel, set(ids), r2num



def diff_relation(idea_kb, back_kb, e2c):
    diff_rel = []
    ids = []
    for head in idea_kb['relations']:
        if e2c[head] in back_kb['relations']:
            for tail in idea_kb['relations'][head]:
                if e2c[tail] in back_kb['relations'][e2c[head]]:
                    for rel in idea_kb['relations'][head][tail]:
                        if rel not in back_kb['relations'][e2c[head]][e2c[tail]]:
                            diff_rel.append((head, tail, rel))
                            ids.extend(idea_kb['relation2sent'][head][tail][rel])

                else:
                    for rel in idea_kb['relations'][head][tail]:
                        diff_rel.append((head, tail, rel))
                        ids.extend(idea_kb['relation2sent'][head][tail][rel])
        else:
            for tail in idea_kb['relations'][head]:
                for rel in idea_kb['relations'][head][tail]:
                    diff_rel.append((head, tail, rel))
                    ids.extend(idea_kb['relation2sent'][head][tail][rel])
    return diff_rel, set(ids)

def get_refer(data):
    refer = []
    for paper in data["metadata"]["references"]:
        refer.append(paper['title'].lower())
    return refer

def related_diff_relation(idea_kb, related_kb, e2c):
    diff_rel = defaultdict(dict)
    same_rel = defaultdict(dict)
    for head in idea_kb['relations']:
        if e2c[head] in related_kb['relations']:
            for tail in idea_kb['relations'][head]:
                if e2c[tail] in related_kb['relations'][e2c[head]]:
                    tmp = []
                    same_tmp = []
                    for rel in idea_kb['relations'][head][tail]:
                        if rel not in related_kb['relations'][e2c[head]][e2c[tail]]:
                            tmp.append(rel)
                        else:
                            same_tmp.append(rel)
                    if len(tmp) > 0:
                        diff_rel[head][tail] = tmp
                    if len(same_tmp) > 0:
                        same_rel[head][tail] = tmp
                else:
                    diff_rel[head][tail] = deepcopy(idea_kb['relations'][head][tail])
        else:
            diff_rel[head] = deepcopy(idea_kb['relations'][head])
    return diff_rel, same_rel

def get_appropriate(back_kb, total_kb, abstract):
    count = 0
    for max_tag in total_kb['cluster2entity']:
        if max_tag in back_kb['entity2cluster']:
            count += 1
        else:
            for cur_tag in total_kb['cluster2entity'][max_tag]:
                if cur_tag in back_kb['entity2cluster']:
                    count += 1
                    break
    return count, len(total_kb['cluster2entity']), abstract

def get_related(back_kb, id2paper, idea_kb, related_kb, key2id, id_, conference, title, e2c, abstract, abstract_list, t_e2c, t_c2t, c2n):
    r2num = {}
    ids = []
    if len(related_kb['entity2cluster']) != 0:
        diff_c, same_c, _, c2type, c2num = diff_cluster(idea_kb, related_kb, t_e2c, t_c2t, c2n)
        diff_rel, same_rel= related_diff_relation(idea_kb, related_kb, e2c)
        diff_kb = idea_kb
        for e in related_kb['cluster2sent']:
            ids.extend(related_kb['cluster2sent'][e])
        for h in related_kb['relation2sent']:
            for t in related_kb['relation2sent'][h]:
                for r in related_kb['relation2sent'][h][t]:
                    ids.extend(related_kb['relation2sent'][h][t][r])
        txts, txts_list = get_sentence(ids, related_kb['sent'])
    else:
        c2num = {}
        same_c = {}
        diff_c = idea_kb['cluster2entity']
        diff_rel = idea_kb['relations']
        c2type = {}
        for c in diff_c:
            if c in t_e2c:
                max_tag = t_e2c[c]
                c2type[c] = t_c2t[max_tag]
                count = 0
                c2num[c] = c2n[t_e2c[c]]
            else:
                type_c = Counter()
                count = 0
                for e in diff_c[c]:
                    if e in idea_kb['entity2type']:
                        type_c[idea_kb['entity2type'][e]] += 1
                    if e in idea_kb['entity2num']:
                        count += idea_kb['entity2num']
                if c in t_e2c:
                    c2num[c] = c2n[t_e2c[c]]
                elif count == 0:
                    c2num[c] = 1
                else:
                    c2num[c] = count
                if len(type_c) == 0:
                    c2type[c] = "OtherScientificTerm"
                else:
                    c2type[c] = type_c.most_common(1)[0][0]
        diff_kb = idea_kb
        same_rel = defaultdict(dict)

    ids_idea = []	
    shared_c = []	
    shared_rel = []	
    paper_ids = set()	
    tmp2id = defaultdict(dict)	
    same_tmp2id = defaultdict(dict)	
    new_e2c = deepcopy(idea_kb['entity2cluster'])	
    for max_tag in same_c:	
        max_ntag = ''	
        if max_tag in back_kb['entity2cluster']:	
            max_ntag = back_kb['entity2cluster'][max_tag]	
            paper_ids |= key2id['cluster'][max_ntag]	
            same_tmp2id['cluster'][max_tag] = list(key2id['cluster'][max_ntag])	
        else:	
            num = 0	
            max_ntag = ''	
            for cur_tag in same_c[max_tag]:	
                if cur_tag in back_kb['entity2cluster']:	
                    if num < len(back_kb['cluster2entity'][back_kb['entity2cluster'][cur_tag]]):	
                        num = len(back_kb['cluster2entity'][back_kb['entity2cluster'][cur_tag]])	
                        max_ntag = back_kb['entity2cluster'][cur_tag]	
            if num > 0:	
                paper_ids |= key2id['cluster'][max_ntag]	
                same_tmp2id['cluster'][max_tag] = list(key2id['cluster'][max_ntag])	
    for max_tag in diff_c:	
        max_ntag = ''	
        if max_tag in back_kb['entity2cluster']:	
            shared_c.append(max_tag)	
            max_ntag = back_kb['entity2cluster'][max_tag]	
            ids_idea.extend(idea_kb['cluster2sent'][max_tag])	
            paper_ids |= key2id['cluster'][max_ntag]	
            tmp2id['cluster'][max_tag] = list(key2id['cluster'][max_ntag])	
        else:	
            num = 0	
            max_ntag = ''	
            for cur_tag in diff_c[max_tag]:	
                if cur_tag in back_kb['entity2cluster']:	
                    if num < len(back_kb['cluster2entity'][back_kb['entity2cluster'][cur_tag]]):	
                        num = len(back_kb['cluster2entity'][back_kb['entity2cluster'][cur_tag]])	
                        max_ntag = back_kb['entity2cluster'][cur_tag]	
            if num > 0:	
                shared_c.append(max_tag)	
                ids_idea.extend(idea_kb['cluster2sent'][max_tag])	
                paper_ids |= key2id['cluster'][max_ntag]	
                tmp2id['cluster'][max_tag] = list(key2id['cluster'][max_ntag])	
        if max_ntag != '':	
            flag = True	
            if max_tag in c2type:	
                c2type[max_ntag] = c2type[max_tag]	
                flag  = False	
            for cur_tag in diff_c[max_tag]:	
                new_e2c[cur_tag] = max_ntag	
                if flag and cur_tag in c2type:	
                    c2type[max_ntag] = c2type[cur_tag]	
            if flag and max_ntag not in c2type:	
                c2type[max_ntag] = "OtherScientificTerm"	
    for head in diff_rel:	
        if new_e2c[head] in back_kb['relations']:	
            for tail in diff_rel[head]:	
                if new_e2c[tail] in back_kb['relations'][new_e2c[head]]:	
                    for rel in diff_rel[head][tail]:	
                        if rel  in back_kb['relations'][new_e2c[head]][new_e2c[tail]]:	
                            shared_rel.append((head, tail, rel))	
                            ids_idea.extend(idea_kb['relation2sent'][head][tail][rel])	
                            paper_ids |= key2id['relation'][new_e2c[head]][new_e2c[tail]][rel]	
                            tmp2id['relation'][str((head, tail, rel))] = list(key2id['relation'][new_e2c[head]][new_e2c[tail]][rel])	
                            r2num[str((head, tail, rel))] = diff_kb['relation2num'][head][tail][rel]	
    for head in same_rel:	
        if new_e2c[head] in back_kb['relations']:	
            for tail in same_rel[head]:	
                if new_e2c[tail] in back_kb['relations'][new_e2c[head]]:	
                    for rel in same_rel[head][tail]:	
                        if rel  in back_kb['relations'][new_e2c[head]][new_e2c[tail]]:		
                            paper_ids |= key2id['relation'][new_e2c[head]][new_e2c[tail]][rel]	
                            same_tmp2id['relation'][str((head, tail, rel))] = list(key2id['relation'][new_e2c[head]][new_e2c[tail]][rel])	
                            r2num[str((head, tail, rel))] = diff_kb['relation2num'][head][tail][rel]	
    paper_ids = sorted(list(paper_ids), reverse=True)	
    with open(conference + id_ + '.pdf.json') as refer:	
        data = json.load(refer)	
    references = get_refer(data)	
    references.append(title.lower())	
    count = 0	
    used = []	
    id2paper_t = {}	
    for pid in paper_ids:	
        if id2paper[pid].lower() not in references:	
            used.append(pid)	
            id2paper_t[pid] = id2paper[pid]	
            count += 1		


    diff_titles = set()
    same_titles = set()
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
        if len(titles) > 0:
            if flag or s >=0.5:
                for t in titles:
                    diff_titles.add(t.lower())
                flag = False

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
            if len(titles) > 0:
                if s >=0.5:
                    for t in titles:
                        diff_titles.add(t.lower())

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
            if len(titles) > 0:
                if flag or s >=0.5:
                    for t in titles:
                        same_titles.add(t.lower())
                    flag = False

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
        if len(titles) > 0:
            if s >=0.5:
                for t in titles:
                    same_titles.add(t.lower())

    if len(ids) == 0:
        if len(ids_idea) != 0:
            txts, txts_list = get_sentence(ids_idea, idea_kb['sent'])
            return (len(diff_titles), len(same_titles), txts)
        return (len(diff_titles), len(same_titles), abstract)
    else:  
        return (len(diff_titles), len(same_titles), txts)
        
def get_sound(contrib_kb, abstract):
    return contrib_kb['total'], contrib_kb['covered'], abstract

def get_novelty(back_kb, idea_kb, abstract, abstract_list, t_e2c, t_c2t, c2n):
    diff_c, ids1, c2t, c2num, e2c = diff_cluster_b(idea_kb, back_kb, t_e2c, t_c2t, c2n)
    diff_rel, ids2, r2num = diff_relation_b(idea_kb, back_kb, e2c)
    sents, txts_list = get_sentence(ids1|ids2, idea_kb['sent'])
    if len(sents) == 0:
        sents = abstract
        txts_list = abstract_list
    new_rel = set()
    new_c = set()
    for h, t, r in diff_rel:
        if r == "USED-FOR":
            try:
                if c2t[t] in ["Method", "Material", "Metric"] and  c2t[h]  == 'Task':
                    new_rel.add((h,t,r))
            except:
                new_rel.add((h,t,r))
        elif r in ['COMPARE', 'FEATURE-OF', "EVALUATE-FOR"]:
            new_rel.add((h,t,r))
    for e in diff_c:
        if e not in c2t:
            new_c.add(e)
        elif c2t[e] in ["Method", "Material", "Task"]:
            new_c.add(e)
        elif 'OtherScientificTerm' == c2t[e] :
            new_c.add(e)

    return (len(new_c), len(new_rel), sents)

def get_impact(future_kb, back_kb, abstract):
    # need merge
    ids = []
    count = 0 
    count_task = 0
    for max_tag in future_kb['cluster2entity']:
        flag = True
        if max_tag in back_kb['entity2cluster']:
            continue
        else:
            for cur_tag in future_kb['cluster2entity'][max_tag]:
                if cur_tag in back_kb['entity2cluster']:
                    flag = False
                    break
        if flag:
            count += 1
            if future_kb['cluster2type'][max_tag] == 'Task':
                count_task += 1
    return count, count_task, abstract

def get_abstract(idea_kb):
    abstract = []
    abstract_list = []
    num = idea_kb['abs_num']
    for sen in idea_kb['sent'][:num]:
        abstract.extend(sen)
        abstract_list.append(sen)
    return abstract, abstract_list

def get_features(back_kb, id2paper, related_kb, idea_kb, future_kb, key2id, index, id_, conference, title, contrib_kb):
    total_kb = merge_cluster(idea_kb, related_kb)
    abstract, abstract_list = get_abstract(idea_kb)
    if index == 'APPROPRIATENESS':
        return get_appropriate(back_kb, total_kb, abstract)
    elif index == 'MEANINGFUL_COMPARISON':
        return get_related(back_kb, id2paper, idea_kb, related_kb, key2id, id_, conference, title, total_kb['idea_e2c'], abstract, abstract_list, total_kb['entity2cluster'], total_kb['cluster2type'], total_kb['cluster2num'])
    elif index == 'SOUNDNESS_CORRECTNESS':
        return get_sound(contrib_kb, abstract)
    elif index == 'ORIGINALITY':
        return get_novelty(back_kb, idea_kb, abstract, abstract_list, total_kb['entity2cluster'], total_kb['cluster2type'], total_kb['cluster2num'])
    elif index == 'IMPACT':
        return get_impact(future_kb, back_kb, abstract)
    elif index == 'CLARITY':
        return abstract




acl_dev = ['37', '94', '173', '352', '371', '489', '660']
acl_test = ['49', '148', '323', '355', '435', '496', '768']
iclr_dev = ['533', '316', '325', '328', '340', '350', '355', '356', '366', '375', '377', '378', '383', '419', '448', '462', '478', '484', '496', '517', '537', '564', '580', '590', '598', '614', '621', '657', '663', '673', '682', '684', '689', '728', '732', '733', '734', '738', '775', '791']
iclr_test = ['330', '333', '358', '363', '398', '400', '412', '438', '444', '457', '460', '471', '482', '486', '498', '518', '554', '556', '566', '574', '597', '611', '612', '632', '648', '670', '678', '687', '691', '697', '719', '739', '745', '749', '756', '767', '773', '778']
acl_aspects = ['RECOMMENDATION', 'APPROPRIATENESS','MEANINGFUL_COMPARISON','SOUNDNESS_CORRECTNESS','ORIGINALITY', 'IMPACT','CLARITY'] 
filename = '../dataset/KGs/back_kg/2016.pkl'
with open(filename, 'rb') as f:
    back_kb = pickle.load(f)

filename = '../dataset/KGs/back_kg/2016_key.pkl'
with open(filename, 'rb') as f:
    key2id = pickle.load(f)

filename = '../dataset/KGs/back_kg/2016_paper.pkl'
with open(filename, 'rb') as f:
    id2paper = pickle.load(f)

a2id = {}
for i, a in enumerate(acl_aspects):
    a2id[a] = i
# load review
reviews = {}
acl_metadata_path = '../dataset/Raw_data/Paper-review Corpus/acl_2017/metadata.txt'
with open(acl_metadata_path, 'r') as f:
    for line in tqdm(f):
        data = json.loads(line)
        id_ = data['id']
        score_mean = {}
        score_avg = {}
        tmp = [ [] for _ in range(7) ]
        for r in data['review']:
            for a in acl_aspects:
                if a in r:
                    tmp[a2id[a]].append(r[a])
                else:
                    tmp[a2id[a]].append(0)
        for i in range(7):
            need = [j for j in tmp[i] if j!=0]
            if len(need) == 0:
                need = [0]
            score_avg[acl_aspects[i]] = statistics.mean(need)
            score_mean[acl_aspects[i]] = round(score_avg[acl_aspects[i]]) #math.floor
        reviews[id_] = {'score':score_mean, 'score_avg':score_avg, 'title':data['paper_content']["title"]}

idea_kbs = {}
whole_ids = []
# load idea_kb
acl_idea_path = '../dataset/KGs/idea_kg/acl_2017.json'
with open(acl_idea_path, 'r') as f:
    for line in tqdm(f):
        data = json.loads(line)
        id_ = data['id']
        whole_ids.append(id_)
        idea_kbs[id_] = data


related_kbs = {}
# load related_kb
acl_relate_path = '../dataset/KGs/related_kg/acl_2017.json'
with open(acl_relate_path, 'r') as f:
    for line in tqdm(f):
        data = json.loads(line)
        id_ = data['id']
        related_kbs[id_] = data

future_kbs = {}
# load future_kb
acl_future_path = '../dataset/KGs/future_kg/acl_2017.json'
with open(acl_future_path, 'r') as f:
    for line in tqdm(f):
        data = json.loads(line)
        id_ = data['id']
        future_kbs[id_] = data
assert len(idea_kbs) == len(related_kbs)

contrib_kbs = {}
# load contrib_kbs
acl_contrib_path = '../dataset/KGs/contribute_kg/acl_2017.json'
with open(acl_contrib_path, 'r') as f:
    for line in tqdm(f):
        data = json.loads(line)
        id_ = data['id']
        contrib_kbs[id_] = data

train_path = 'acl_2017/%s_train.json'
valid_path = 'acl_2017/%s_valid.json'
test_path = 'acl_2017/%s_test.json'
for id_ in tqdm(whole_ids):
    idea_kb = idea_kbs[id_]
    related_kb = related_kbs[id_]
    future_kb = future_kbs[id_]
    contrib_kb = contrib_kbs[id_]
    scores = reviews[id_]['score']
    scores_avgs = reviews[id_]['score_avg']
    title = reviews[id_]['title']
    if id_ in acl_dev:
        cur_path = valid_path
    elif id_ in acl_test:
        cur_path = test_path
    else:
        cur_path = train_path
    features = {}
    for a in acl_aspects[1:]:
        features[a] = get_features(back_kb, id2paper, related_kb, idea_kb, future_kb, key2id, a, id_, '../dataset/Raw_data/Paper-review Corpus/acl_2017/txt/', title, contrib_kb)
        score = scores[a]
        score_avg = scores_avgs[a]
        if score != 0:
            tmp = {
                'id': 'acl_2017-' + id_, 
                'feature': features[a],
                'score': score,
                'score_avg': score_avg
            }
            with open(cur_path % a, 'a') as f:
                f.write(json.dumps(tmp)+'\n')
    score = scores['RECOMMENDATION']
    score_avg = scores_avgs['RECOMMENDATION']
    if score != 0:
        tmp = {
            'id': 'acl_2017-' + id_, 
            'feature': features,
            'score': score,
            'score_avg': score_avg
        }
        with open(cur_path % 'RECOMMENDATION', 'a') as f:
            f.write(json.dumps(tmp)+'\n')


