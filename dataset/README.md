# ReviewRobot Dataset

## Overview
This repository contains data for paper ReviewRobot: Explainable Paper Review Generation based on Knowledge Synthesis. [[Dataset]](https://drive.google.com/file/d/1NclEwGEVcHCrSWk8s3lDjvEbMlWXQoXM/view?usp=sharing)

## Dataset
There are three folders: `Raw_data`, `IE_result`, and `KGs`. 

### Raw_data folder

The `Raw_data` has two parts: `Background` Corpus and `Paper-review` Corpus. 

We create the `Background Corpus` by selecting machine learning related pappers from the [Semantic Scholar Open Research Corpus](http://s2-public-api-prod.us-west-2.elasticbeanstalk.com/corpus/). It contains papers with their titles and abstracts published from the year of 1965 to 2019 (included).

The `Paper-review Corpus` contains parsed paper pdfs and their corresponding reviews. The paper-review pairs of `acl_2017` and `iclr_2017` folders come from [PeerRead dataset](https://github.com/allenai/PeerRead). We fetched the rest from [OpenReview](https://openreview.net/) and [NeruIPS](https://papers.nips.cc/). We parsed those pdfs using [GROBID](https://github.com/kermitt2/grobid). In each folder, `metadata.txt` contains all human reviews, and the `txt/` folder contains all processed papers.


### IE_result folder
The `IE_result` folder contains information extraction results from [SciIE](https://bitbucket.org/luanyi/scierc/src/master/). In each group, the `*_json/` contains tokenized texts, and the `*_output/` contains IE results of tokenized texts.

The `Background_IE` contains two folders from one group for all paper abstracts from 1965 to 2019.

The `Paper-review_IE` contains four folders from two groups. The first group: `iclrnipsabs_json` and `iclrnipsabs_output` contain IE results for abstracts of `Paper-review Corpus`. The second group: `iclrnips_json` and `iclrnips_output` contain IE results for rest of papers in `Paper-review Corpus`.

### KGs
The `KGs` folder contains the knowledge graphs built on the `IE_result`. 

#### back_kg
The `back_kg` contains the background KGs built up to a certain year. For each year, there are three files. 

Take 2012 as an example:
*   `2012.pkl` contains the background knowledge graph up to (include) 2012. It contains a dictionary of 6 fields: `num_doc` is the number of papers up to that year, `cluster2entity` is a mapping from the entity to its mentions, `entity2cluster` is a mapping from the mention to its corresponding entity, `cluster2type` is a mapping from the entity to its type, `entity` refers to all mentions in current KG, and `relations` refers to all relations in current KG. 
*   `2012_key.pkl` contains the mappings from knowledge elements to paper ids. It has two fields: `cluster` is the mapping from an entity to its corresponding paper ids, and `relation` is the mapping from a relation to the corresponding paper ids.
*   `2012_paper` contains the mappings from paper id to its paper title.

#### idea_kg
The `idea_kg` folder contains idea KGs constructed from paper abstracts and conclusions. Each line is a paper in the venue and has the following fields: `id` for the paper id, `abs_num` for the number of abstract sentences, `sent` for all sentences related to  `idea_kg`, `entity` for all mentions in current KG, `cluster2sent` for the corresponding sentence ids for a specific entity, `entity2num` for the occurence of a specific mention, `relation2num` for the occurence of a specific relation, `cluster2entity` for a mapping from the entity to its mentions, `entity2type` conains a mapping from the mention to the type, `relations` for all relations in current KG, `relation2sent` for corresponding sentence ids for a specific relation, and `entity2cluster` for a mapping from the mention to its corresponding entity. 

#### related_kg 
The `related_kg` contains related KGs constructed from related work for each venue. It is of the same structure as `idea_kg`.

#### contribute_kg
The `contribute_kg` contains contribute KGs constructed from paper contribution section (under introduction section) and experiment section. It contains a dictionary of 4 fields:  `id` for the paper id,  `total` for the number of entities covered in the contribution section, `covered` for the number of entities covered in the experiment section, `sents` related sentences that covered those entities from both sections.

#### future_kg
The `future_kg` contains future KGs constructed from future work for each venue. It is of the same structure as `idea_kg`.

### Review-annotation
The `Review-annotation` folder contains human annotations for review category and paper-review sentence pairs. The `review.txt` contains annotation for review category including 236 sentences for "SUMMARY", 33 sentences for "NOVELTY", 174 sentences for "SOUNDNESS_CORRECTNESS", 16 sentences for "MEANINGFUL_COMPARISON", and 14 sentences for "IMPACT". The `pair.txt`   contains 2,535 review-paper pairs. For each pair, the first slot is the review sentence; the second slot is the paper sentence, the third slot is the label where 0 indicates two sentences are not related and 1 indicates they are related.

## License
Creative Commons — Attribution 4.0 International — CC BY 4.0