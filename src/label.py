import spacy
import pandas as pd
import re
from datetime import datetime as dt
from spacy.tokenizer import _get_regex_pattern

def load(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        tweets_str = [[line.rstrip('\n')] for line in f]
    tweets = pd.DataFrame.from_records(tweets_str,columns=['raw'])
    return tweets

def label(texts):
    tweets = pd.DataFrame(columns=['tweet'])

    # load model and add hashtag/hypen regex patterns
    nlp = spacy.load('en_core_web_md')
    nlp.add_pipe(nlp.create_pipe('sentencizer'), before='parser')
    re_token_match = _get_regex_pattern(nlp.Defaults.token_match)
    re_token_match = f"({re_token_match}|#\w+|\w+-\w+)"
    nlp.tokenizer.token_match = re.compile(re_token_match).match

    docs = nlp.pipe(texts,disable=['tagger','textcat'])
    for i,doc in enumerate(docs):
        tweet_header = '#{}\n#{}\n'.format(i,doc.text)
        sentences = []
        for j,sent in enumerate(doc.sents):
            sentence_header = '#{}\n#{}\n'.format(j,sent.text)
            sentence_string = '\n'.join(['{}\t{}\t{}'.format(i+1,token.text,token.ent_iob_+'-'+token.ent_type_ if token.ent_type_ else 'O') for i,token in enumerate(sent)])
            sentences.append(sentence_header+sentence_string)
        tweet = '\n\n'.join(sentences)
        tweets = tweets.append({'tweet':tweet_header+tweet+'\n'},ignore_index=True)

    return tweets

def write(tweets,filename='../data/'+dt.now().strftime("%d_%m_%y")+'_labelled.txt'):
    try:
        with open(filename,'wb') as f:
            for _,tweet in tweets.items():
                f.write((tweet+'\n').encode('utf-8'))
        return True
    except Exception as e:
        print(e)
        return False
    

if __name__=='__main__':
    tweets = load('../data/16_10_19.txt')
    labelled_tweets = label(tweets['raw'])
    write(labelled_tweets['tweet'])