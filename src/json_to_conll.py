import json
import pandas as pd
import sys
from datetime import datetime as dt
import os


'''
    Doccano only returns json format, but we want to work with conll format. That's where this script comes in.
'''


def load(filename):
    ''' load json returned from doccano, one row per entry '''

    with open(filename, 'r', encoding='utf-8') as f:
        tweets = {i: json.loads(line) for i, line in enumerate(f)}
    return tweets


def label(items):
    '''Takes dict of dicts and returns series of strings
    in conll format, ready to be written
    Note: \# is changed back to # (so if you load it into doccano again it will disappear)
    '''

    tweets = []
    for item in items.values():
        text = item['text']
        labels = sorted(item['labels'], key=lambda x: x[0])
        x = [text[a:b] for a, b, _ in labels]
        y = [c for _, _, c in labels]
        tweet_string = '\n'.join(['{}\t{}'.format(
            token if token != '\#' else '#', tag) for token, tag in zip(x, y)])
        tweets.append(tweet_string + '\n')
    return pd.Series(tweets)


def write(tweets, filename='../data/' + dt.now().strftime("%d_%m_%y") + '_doccano_labelled.txt'):
    ''' take strings in conll format and save them to file '''

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
    labelled_tweets = label(tweets)
    write(labelled_tweets)
