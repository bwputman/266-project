from __future__ import division

import nltk
import numpy as np
import pandas as pd
import scipy 
import re, os, sys

import spacy
import seaborn as sns
import matplotlib.pyplot as plt

from subject_object_extraction import findSVOs

import matplotlib
matplotlib.style.use('ggplot')

print 'loading training data'
training_data = pd.read_csv('./data/train.csv', nrows = 100, encoding = 'utf-8').fillna("")
df_train = training_data.copy()

print 'loading testing data'
testing_data  = pd.read_csv('./data/test.csv', nrows = 100, encoding = 'utf-8').fillna("")
df_test = testing_data.copy()

# Cleaning data, remove leading and tailing spaces
df_train['q1'] = df_train['question1'].map(lambda q: re.sub("\s\s+" , " ", q)) 
df_train['q2'] = df_train['question2'].map(lambda q: re.sub("\s\s+" , " ", q)) 

df_test['q1'] = df_train['question1'].map(lambda q: re.sub("\s\s+" , " ", q)) 
df_test['q2'] = df_train['question2'].map(lambda q: re.sub("\s\s+" , " ", q)) 

print 'loading spacy model en_core_web_md'
nlp = spacy.load('en_core_web_md')
nlp.vocab.add_flag(lambda s: s.lower() in spacy.en.word_sets.STOP_WORDS, spacy.attrs.IS_STOP)


def nlp_parse(q1, q2 = None):
    token1 = []
    lemma1 = []
    pos1 = []
    tag1 =[]
    dep1 = []
#     shape = []
    alpha1 = []
    stop1 =[]
    doc1 = nlp(unicode(q1))
    for w in doc1:
        token1.append(w.text)
        lemma1.append(w.lemma_)
        pos1.append(w.pos_)
        tag1.append(w.tag_)
        dep1.append(w.dep_)
#         shape1.append(w.shape_)
        alpha1.append(w.is_alpha)
        stop1.append(w.is_stop)
    word_cnt1 = len(token1)
    svo1 = findSVOs(doc1)
    ents1 = [ (e.label_, e.text) for e in doc1.ents]
    alpha1_cnt = sum(alpha1)
    stop1_cnt = sum(stop1)   
    svo1_len = len(svo1)
    if q2 is None:
        return token1, lemma1, pos1, tag1, dep1, stop1, word_cnt1, svo1, ents1, alpha1_cnt, stop1_cnt, svo1_len
    
    doc2 = nlp(unicode(q2))
    doc_similarity = doc1.similarity(doc2)
    
    token2 = []
    lemma2 = []
    pos2 = []
    tag2 =[]
    dep2 = []
#     shape2 = []     
    alpha2 = []
    stop2 = []
    for w in doc2:
        token2.append(w.text)
        lemma2.append(w.lemma_)
        pos2.append(w.pos_)
        tag2.append(w.tag_)
        dep2.append(w.dep_)
#         shape2.append(w.shape_)
        alpha2.append(w.is_alpha)
        stop2.append(w.is_stop)
    word_cnt2 = len(token2)
    svo2 = findSVOs(doc2)
    ents2 = [ (e.label_, e.text) for e in doc2.ents]

    alpha2_cnt = sum(alpha2)
    stop1_cnt = sum(stop2)
    svo1_len = len(svo2)
    
    return  token1, lemma1, pos1, tag1, dep1, stop1, word_cnt1, svo1, ents1, alpha1_cnt, stop1_cnt, svo1_len, \
                token2, lemma2, pos2, tag2, dep2, stop2, word_cnt2, svo2, ents2, alpha1_cnt, stop1_cnt, svo1_len, \
                doc_similarity

use_parse_1 = False

if use_parse_1:
    print 'parse 1: ready to process for df_train'
    
    df_train['token1'], df_train['lemma1'], df_train['pos1'], \
    df_train['tag1'], df_train['dep1'], df_train['stop1'], \
    df_train['word_cnt1'], df_train['svo1'], df_train['ents1'], \
    df_train['alpha1_cnt'], df_train['stop1_cnt'], df_train['svo1_len'], \
    df_train['token2'], df_train['lemma2'], df_train['pos2'], \
    df_train['tag2'], df_train['dep2'], df_train['stop2'], \
    df_train['word_cnt2'], df_train['svo2'], df_train['ents2'], \
    df_train['alpha2_cnt'], df_train['stop2_cnt'], df_train['svo2_len'], \
    df_train['doc_sim'] = zip( *df_train.apply(lambda df: nlp_parse(df['q1'], df['q2']), axis=1)) 
    
    df_train.to_pickle("./data/df_train.pkl")
    
    print 'ready to process for df_test'
    
    df_test['token1'], df_test['lemma1'], df_test['pos1'], \
    df_test['tag1'], df_test['dep1'], df_test['stop1'], \
    df_test['word_cnt1'], df_test['svo1'], df_test['ents1'], \
    df_test['alpha1_cnt'], df_test['stop1_cnt'], df_test['svo1_len'], \
    df_test['token2'], df_test['lemma2'], df_test['pos2'], \
    df_test['tag2'], df_test['dep2'], df_test['stop2'], \
    df_test['word_cnt2'], df_test['svo2'], df_test['ents2'], \
    df_test['alpha2_cnt'], df_test['stop2_cnt'], df_test['svo2_len'], \
    df_test['doc_sim'] = zip( *df_test.apply(lambda df: nlp_parse(df['q1'], df['q2']), axis=1)) 
    
    df_test.to_pickle("./data/df_test.pkl")
    
    print 'done'

def nlp_parse2(q1, q2 = None):
    token1 = []
    lemma1 = []
    pos1 = []
    tag1 =[]
    dep1 = []
#     shape = []
    alpha1 = []
    stop1 =[]
    doc1 = nlp(unicode(q1))
    for w in doc1:
        token1.append(w.text)
        lemma1.append(w.lemma_)
        pos1.append(w.pos_)
        tag1.append(w.tag_)
        dep1.append(w.dep_)
#         shape1.append(w.shape_)
        alpha1.append(w.is_alpha)
        stop1.append(w.is_stop)
    word_cnt1 = len(token1)
    svo1 = findSVOs(doc1)
    ents1 = [ (e.label_, e.text) for e in doc1.ents]
    alpha1_cnt = sum(alpha1)
    stop1_cnt = sum(stop1)   
    svo1_cnt = len(svo1)
    #svo_l = [" ".join(svo) for svo in svo1] # convert the svo
    #svo_str1 = " , ".join(svo_l) 
    if q2 is None:
        return " ".join(token1), " ".join(lemma1), " ".join(pos1), " ".join(tag1), " ".join(dep1), \
               word_cnt1, svo1, ents1, alpha1_cnt, stop1_cnt, svo1_cnt
    
    doc2 = nlp(unicode(q2))
    doc_similarity = doc1.similarity(doc2)
    
    token2 = []
    lemma2 = []
    pos2 = []
    tag2 =[]
    dep2 = []
#     shape2 = []     
    alpha2 = []
    stop2 = []
    for w in doc2:
        token2.append(w.text)
        lemma2.append(w.lemma_)
        pos2.append(w.pos_)
        tag2.append(w.tag_)
        dep2.append(w.dep_)
#         shape2.append(w.shape_)
        alpha2.append(w.is_alpha)
        stop2.append(w.is_stop)
    word_cnt2 = len(token2)
    svo2 = findSVOs(doc2)
    ents2 = [ (e.label_, e.text) for e in doc2.ents]

    alpha2_cnt = sum(alpha2)
    stop2_cnt = sum(stop2)
    svo2_cnt = len(svo2)
    #svo_l = [" ".join(svo) for svo in svo2] # convert the svo
    #svo_str2 = " , ".join(svo_l)     
    return  " ".join(token1), " ".join(lemma1), " ".join(pos1), " ".join(tag1), " ".join(dep1), \
            word_cnt1, svo1, ents1, alpha1_cnt, stop1_cnt, svo1_cnt, \
            " ".join(token2), " ".join(lemma2), " ".join(pos2), " ".join(tag2), " ".join(dep2), \
            word_cnt2, svo2, ents2, alpha2_cnt, stop2_cnt, svo2_cnt, \
            doc_similarity

use_parse_2 = True
if use_parse_2:
    print 'parse 1: ready to process for df_train'
    
    df_train['token1'], df_train['lemma1'], df_train['pos1'], \
    df_train['tag1'], df_train['dep1'], df_train['word_cnt1'], \
    df_train['svo1'], df_train['ents1'], df_train['alpha1_cnt'], \
    df_train['stop1_cnt'], df_train['svo1_cnt'],  \
    df_train['token2'], df_train['lemma2'], df_train['pos2'], \
    df_train['tag2'], df_train['dep2'], df_train['word_cnt2'], \
    df_train['svo2'], df_train['ents2'], df_train['alpha2_cnt'], \
    df_train['stop2_cnt'], df_train['svo2_cnt'], \
    df_train['doc_sim'] = zip( *df_train.apply(lambda df: nlp_parse2(df['q1'], df['q2']), axis=1)) 
    
    df_train.to_pickle("./data/df_train.pkl")
    
    print 'ready to process for df_test'
    
    df_test['token1'], df_test['lemma1'], df_test['pos1'], \
    df_test['tag1'], df_test['dep1'], df_test['word_cnt1'], \
    df_test['svo1'], df_test['ents1'], df_test['alpha1_cnt'], \
    df_test['stop1_cnt'], df_test['svo1_cnt'],  \
    df_test['token2'], df_test['lemma2'], df_test['pos2'], \
    df_test['tag2'], df_test['dep2'], df_test['word_cnt2'], \
    df_test['svo2'], df_test['ents2'], df_test['alpha2_cnt'], \
    df_test['stop2_cnt'], df_test['svo2_cnt'], \
    df_test['doc_sim'] = zip( *df_test.apply(lambda df: nlp_parse2(df['q1'], df['q2']), axis=1)) 
    
    df_test.to_pickle("./data/df_test.pkl")
    
    print 'done'    
    