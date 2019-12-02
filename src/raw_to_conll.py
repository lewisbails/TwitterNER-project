import spacy
import pandas as pd
import re
import sys
from datetime import datetime as dt
from spacy.tokenizer import _get_regex_pattern
import os


def load(filename):
    ''' get series of raw tweets '''
    with open(filename, 'r', encoding='utf-8') as f:
        tweets_str = [line.rstrip('\n') for line in f]
    return pd.Series(tweets_str)


def tag(token):
    ''' Returns tag of token '''
    if token.ent_type_:
        return token.ent_iob_ + '-' + token.ent_type_
    else:
        return 'O'


def get_docs(texts):
    ''' Returns a doc generator from raw tweet strings'''

    nlp = spacy.load('en_core_web_lg')

    re_token_match = _get_regex_pattern(nlp.Defaults.token_match)
    re_token_match = f"({re_token_match}|\w+-\w+)"
    nlp.tokenizer.token_match = re.compile(re_token_match).match

    docs = nlp.pipe(texts, disable=['textcat'])
    return docs


def to_conll(docs):
    ''' convert the doc objects to strings in conll annotations format '''

    tweets = []
    for i, tweet in enumerate(docs):
        tweet_header = f'#{i}\n#{tweet.text}\n'
        tweet_string = '\n'.join(['{}\t{}'.format(
            token.text if token.text != '#' else '\#', tag(token)) for token in tweet])
        tweets.append(tweet_header + tweet_string + '\n')
    return pd.Series(tweets)


def write(tweets, filename='../data/' + dt.now().strftime("%d_%m_%y") + '_spacy_labelled.txt'):
    ''' write the series of conll formatted tweets to file '''

    mode = 'ab' if os.path.exists(filename) else 'wb'
    try:
        with open(filename, mode) as f:
            for tweet in tweets:
                f.write((tweet + '\n').encode('utf-8'))
        print(f'Saved to {filename}')
        return True
    except Exception as e:
        print(e)
        return False


if __name__ == '__main__':
    assert len(sys.argv) == 2, '1 argument accepted.'

    tweets = load(sys.argv[1])
    docs = get_docs(tweets)
    labelled_tweets = to_conll(docs)
    write(labelled_tweets)

    # from collections import Counter
    # import matplotlib.pyplot as plt
    # from matplotlib.legend_handler import HandlerBase
    # from matplotlib.text import Text
    # import seaborn as sns
    # flat = [token.text for doc in docs for token in doc]

    # labels = pd.Series([token.ent_type_ for doc in docs for token in doc if token.ent_iob_ == 'B']).replace(
    #     '', None).dropna()
    # labels.value_counts(normalize=True).to_csv('vc.csv')

    # tags_of_interest = ['LOC', 'WORK_OF_ART', 'EVENT', 'FAC']
    # records = [[ent.label_, ent.text.lower()]
    #            for doc in docs for ent in doc.ents]
    # records = pd.DataFrame.from_records(records, columns=['tag', 'text']).replace(
    #     '', {'tag': None}).dropna(subset=['tag'])
    # for name, group in records.groupby('tag'):
    #     if name in tags_of_interest:
    #         ax = group['text'].head(50).value_counts(
    #             normalize=True).plot(kind='bar')
    #         ax.set_title(f'PMF - {name}')
    #         plt.xticks(rotation=70, ha='right')
    #         plt.show()

    # labels = pd.Series(['ENTITY' if token.ent_type_ else 'O' for doc in docs for token in doc])
    # ax = labels.value_counts(normalize=True).plot(kind='bar')
    # ax.set_title('Probability Mass Function - Entities')
    # plt.xticks(rotation=45)
    # plt.show()
