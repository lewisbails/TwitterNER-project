import spacy
import pandas as pd
import re
import sys
from datetime import datetime as dt
from spacy.tokenizer import _get_regex_pattern
import os

def load(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        tweets_str = [[line.rstrip('\n')] for line in f]
    tweets = pd.DataFrame.from_records(tweets_str,columns=['raw'])
    return tweets

def tag(token):
    # predifined entities
    entities = ['PERSON','NORP','ORG','GPE','LOC','PRODUCT','WORK_OF_ART','EVENT','FAC']

    if token.ent_type_ and token.ent_type_ in entities and not token.text.startswith('@'):
        return token.ent_iob_+'-'+token.ent_type_
    else:
        return 'O'

def label(texts):
    tweets = pd.DataFrame(columns=['tweet'])

    nlp = spacy.load('en_core_web_lg')
    re_token_match = _get_regex_pattern(nlp.Defaults.token_match)
    # re_token_match = f"({re_token_match}|#\w+|\w+-\w+)" #hashtag too?
    re_token_match = f"({re_token_match}|\w+-\w+)"
    nlp.tokenizer.token_match = re.compile(re_token_match).match

    docs = nlp.pipe(texts,disable=['tagger','textcat'])
    for i,tweet in enumerate(docs):
        tweet_header = f'#{i}\n#{tweet.text}\n'
        tweet_string = '\n'.join(['{}\t{}'.format(token.text if token.text!='#' else '\#',tag(token)) for token in tweet])
        tweets = tweets.append({'tweet':tweet_header+tweet_string+'\n'},ignore_index=True)

    return tweets,nlp.pipe(texts,disable=['tagger','textcat'])

def write(tweets,filename='../data/'+dt.now().strftime("%d_%m_%y")+'_spacy_labelled.txt'):
    mode = 'ab' if os.path.exists(filename) else 'wb'
    try:
        with open(filename,mode) as f:
            for _,tweet in tweets.items():
                f.write((tweet+'\n').encode('utf-8'))
        print(f'Saved to {filename}')
        return True
    except Exception as e:
        print(e)
        return False
    

if __name__=='__main__':
    assert len(sys.argv) == 2, '1 argument accepted.'

    tweets = load(sys.argv[1])
    labelled_tweets,_ = label(tweets['raw'])
    write(labelled_tweets['tweet'])

    # from collections import Counter
    # flat = [token.text for doc in docs for token in doc]
    # counts = Counter(flat)
    # print(counts.most_common(5))