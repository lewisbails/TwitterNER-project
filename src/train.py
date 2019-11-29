#!/usr/bin/env python
# coding: utf-8

# ##Dependencies

# In[126]:


import logging
logging.basicConfig(filename='train.log', level=logging.DEBUG)
logging.info('Ready to log.')

def log_print(text, level='info'):
    print(text)
    if level=='info':
        logging.info(text)
    elif level=='error':
        logging.error(text)
    else:
        logging.debug(text)


# In[127]:


import scipy
import pandas as pd
import numpy as np
import eli5
from sklearn_crfsuite import metrics, scorers, CRF
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator, TransformerMixin 
import pickle
from datetime import datetime as dt
import spacy
import json
import seaborn as sns
import matplotlib.pyplot as plt
import os

log_print('Imports successful.')


# ##Inspect

# In[128]:


def get_frames(path, k_fold=True):
    data = {i:pd.read_csv(path+f'{i}.csv', index_col=['sentence_id','token_id']).fillna(value={'token': 'NA'}) for i in ('train','validation','test')}

    if k_fold:
        offset = data['train'].index.max(0)[0]+1
        df = data['validation']
        df.index = df.index.set_levels(df.index.levels[0]+offset, level=0)
        data['train'] = pd.concat([data['train'],data['validation']], verify_integrity=True)

    return data


# In[129]:


try:
    path = '../data/' if 'src' in os.getcwd() else './data/'
    data_conll = get_frames(path+'conll/')
    data_wnut = get_frames(path+'wnut/')

    logging.info('Frames loaded from file.')
except Exception as e:
    logging.error(f'Failed: {e}')
    raise e


# In[130]:


tag_map = {
    'B-product':'B-MISC',
    'I-product':'I-MISC',
    'B-creative-work':'B-MISC',
    'I-creative-work':'I-MISC',
    'B-corporation':'B-ORG',
    'I-corporation':'I-ORG',
    'B-group':'B-MISC',
    'I-group':'I-MISC',
    'B-person':'B-PER',
    'I-person':'I-PER',
    'B-location':'B-LOC',
    'I-location':'I-LOC',
    'O':'O'
}


# In[131]:


try:
    for frame in data_wnut.values():
        frame['tag'] = frame['tag'].map(tag_map)

    wnut_sort = sorted(data_conll['train']['tag'].unique(), key=lambda x: (x[1:],x[0]))
    conll_sort = sorted(data_wnut['train']['tag'].unique(), key=lambda x: (x[1:],x[0]))

    assert all([i==j for i,j in zip(wnut_sort,conll_sort)]), 'Mismatched tags between CoNLL and WNUT data.'

    log_print(f'Tag conversion successful. Tags: {str(wnut_sort)}')

except Exception as e:
    logging.error(f'Failed: {e}')
    raise e


# In[132]:


# Get the self-annotated tweets
tweets = pd.read_csv(path+f'twitter/test.csv', index_col=['sentence_id','token_id']).fillna(value={'token': 'NA'})
tweet_map = {
    'B-EVENT':'B-MISC',
    'I-EVENT':'I-MISC',
    'B-NORP':'B-MISC',
    'I-NORP':'I-MISC',
    'B-WORK_OF_ART':'B-MISC',
    'I-WORK_OF_ART':'I-MISC',
    'B-PRODUCT':'B-MISC',
    'I-PRODUCT':'I-MISC',
    'B-FAC':'B-MISC',
    'I-FAC':'I-MISC',
    'B-PERSON':'B-PER',
    'I-PERSON':'I-PER',
    'B-GPE':'B-LOC',
    'I-GPE':'I-LOC',
    'B-LOC':'B-LOC',
    'I-LOC':'I-LOC',
    'B-ORG':'B-ORG',
    'I-ORG':'I-ORG',
    'O':'O'
}

tweets['tag'] = tweets['tag'].map(tweet_map)

tweet_tags = sorted(tweets['tag'].unique(), key=lambda x: (x[1:],x[0]))
log_print(f'Testing tweets ready. Tags: {str(tweet_tags)}')
tweets.head()


# ##Feature Engineering

# Coming up with good features for our model to assign weights to is critical. We've used a number of orthographic features, prefixes, suffixes, POS tags and lemmas. The tag transition is also modelled under the hood by crfsuite. The context window is of radius 1.

# In[133]:


class Sent2():
    '''Takes an array of sentences and their tags'''

    def __init__(self,attributes=None):
        self.attrs = attributes

    def __get_features(self, word, prefix):
        '''get features dictionary for a single token'''

        try:

            features = {
                f'{prefix}{attr}': word[attr] for attr in self.attrs
            }

            features.update({f'{prefix}bias': 1.0,
            f'{prefix}prefix2': word['token'][:2],
            f'{prefix}prefix3': word['token'][:2],
            f'{prefix}suffix2': word['token'][-2:],
            f'{prefix}suffix3': word['token'][-3:],
            })

        except TypeError:
            key = 'BOS' if prefix == '-1' else 'EOS'
            features = {key: True}
        return features

    def __word2features(self, sent, token_id):
        '''get features dictionary over the context window'''

        # get rows of context window
        current = sent.iloc[token_id]
        left = sent.iloc[token_id - 1] if token_id else None
        try:
            right = sent.iloc[token_id + 1]
        except:
            right = None

        features = {}

        # add features from all tokens in context window
        for row, prefix in zip((current, left, right), ('', '-1', '+1')):
            features.update(self.__get_features(row, prefix))

        features.pop('-1bias', None)
        features.pop('+1bias', None)

        return features

    def features(self, sent):
        '''convert a sentence to a list of context window feature vectors'''
        return [self.__word2features(sent, i) for i in range(len(sent))]

    def tokens(self, sent):
        '''get the sequence of string tokens for a sentence'''
        return ' '.join([row['token'] for _, row in sent.iterrows()])

    def labels(self,sent):
        '''get the sequence of target tags for a sentence'''
        return [row['tag'] for _, row in sent.iterrows()]


# Sample sentences from our dataframes and create the input and output pairs for training/testing. An input is a list of feature vectors, one for each time step.

# In[134]:


def get_sentences(frame, p=1, name='unknown'):

    frame = frame.drop(columns='cluster')

    if p<1:
        sample_idx = np.random.choice(range(len(frame.groupby(level='sentence_id'))),
                                size=int(p*len(frame.groupby(level='sentence_id'))),
                                replace=False)
        frame = frame.loc[sample_idx,:]

    att = list(frame.columns.values)
    att.remove('tag')
    # print('Using token features:\n',att)

    x = frame.drop(columns='tag'
                    ).groupby(level='sentence_id'
                    ).apply(lambda sent: Sent2(att).features(sent))
    y = frame.groupby(level='sentence_id'
                    ).apply(lambda sent: Sent2().labels(sent))
    z = frame.groupby(level='sentence_id'
                    ).apply(lambda sent: Sent2().tokens(sent))


    # return pd.DataFrame(data={'x': x, 'y': y, 'tokens': z, 'dist': dist})
    return pd.DataFrame(data={'x': x, 'y': y, 'z': z, 'dist': [name]*len(x)})


# In[135]:


dist = 'mixed'
p = 0.25

try:
    log_print('Getting sentences from multi-index dataframes...')

    test_tweet = get_sentences(tweets, 1, 'tweet')

    if dist == 'mixed':
        train_conll = get_sentences(data_conll['train'], p, 'conll')
        train_wnut = get_sentences(data_wnut['train'], p, 'wnut')
        train = pd.concat([train_conll,train_wnut]).reset_index(drop=True).sample(frac=1)
    elif dist == 'conll':
        train = get_sentences(data_conll['train'], p, 'conll')
    else:
        train = get_sentences(data_wnut['train'], p, 'wnut')

    log_print(f'Conversion to sentences successful. {len(train)} training sentences for {dist} distribution, {len(test_tweet)} tweet test sentences.')
except Exception as e:
    logging.error(f'Failed: {e}')
    raise e


# ##Train

# The 'O' is not of interest to us; we are more interested in the other tags.
# Let's evaluate models based on a flat f1 score over the other tags.

# In[93]:


labels = list(data_conll['train']['tag'].unique())
labels.remove('O')

flat_f1 = make_scorer(metrics.flat_f1_score, average='weighted', labels=labels)


# In[113]:


get_model = 'gs' # gs, fit, load

date = dt.now().strftime('%d_%m_%y')
model_name = f'crf_{date}_{dist}.pickle'

try:
    if get_model == 'gs':
        iters = 20
        cv = 3
        params_space = {
        'c1': scipy.stats.expon(scale=1),
        'c2': scipy.stats.expon(scale=0.02),
        }

        logging.info(f'Fitting {iters*cv} models...')
        rs = RandomizedSearchCV(CRF(all_possible_transitions=True), params_space, n_iter=iters, cv=cv, scoring=flat_f1, n_jobs=10, verbose=1)
        rs.fit(train['x'], train['y'])
        clf = rs.best_estimator_
        log_print(f'Fit successful. Best params {rs.best_params_}')
    elif get_model == 'fit':
        logging.info(f'Fitting full model...')
        clf = CRF(all_possible_transitions=True, c1=0.0798, c2=0.0523)
        clf.fit(train['x'], train['y'])
        log_print(f'Fit successful.')
    else:
        with open(model_name, 'rb') as f:
            clf = pickle.load(f)
        log_print(f'Loaded pre-trained model {model_name}.')
except Exception as e:
    logging.error(f'Failed: {e}')
    raise e


# In[114]:


clf


# In[115]:


if get_model != 'load':
    with open(model_name, 'wb') as f:
        pickle.dump(clf,f)


# ##Evaluate

# In[108]:


y_pred = clf.predict(test_tweet['x'])

flat_f1 = metrics.flat_f1_score(test_tweet['y'], y_pred,
                      average='weighted', labels=labels)

log_print('Flat F1 score: '+str(flat_f1))


# In[123]:


sorted_labels = sorted(labels, key = lambda x: (x[1:],x[0]))

print(metrics.flat_classification_report(
    test_tweet['y'], y_pred, labels=sorted_labels, digits=3
))


# In[124]:


if os.name == 'nt':
    eli5.show_weights(clf, top=10)


# In[125]:


if get_model == 'gs' and os.name == 'nt':
    _x = [s['c1'] for s in rs.cv_results_['params']]
    _y = [s['c2'] for s in rs.cv_results_['params']]
    _c = [round(s,2) for s in rs.cv_results_['mean_test_score']]

    fig = plt.figure()
    fig.set_size_inches(12, 12)
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('C1')
    ax.set_ylabel('C2')
    ax.set_title("Randomized Hyperparameter Search Sampled CV Results (min={:0.3}, max={:0.3})".format(
        min(_c), max(_c)
    ))

    sns.scatterplot(x=_x, y=_y, hue=_c, size=_c, ax=ax)

    # ax.scatter(_x, _y, c=_c, s=60, alpha=0.9, edgecolors=[0,0,0])

    print(f'Dark blue => {round(min(_c),3)}, dark red => {round(max(_c),3)}')

    logging.info('Evaluation successful.')


# In[53]:




