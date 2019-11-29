import json
import pandas as pd
import sys
import spacy
from spacy.tokens import Doc

''' For turning CONLL-2003 data to a CSVs with a multitude of columms'''


columns = ['pos', 'dep', 'lemma', 'norm', 'lower', 'shape', 'is_alpha', 'is_ascii', 'is_digit', 'is_lower',
           'is_upper', 'is_title', 'is_punct', 'like_num', 'is_oov', 'is_stop', 'cluster', 'like_url', 'is_currency']


def json_to_dict(filename):
    '''Get dict of sentences from json1 returned from doccano'''

    with open(filename, 'r', encoding='utf-8') as f:
        sents = {i: json.loads(line) for i, line in enumerate(f)}

    return sents


def get_features(token, attributes):
    features = []
    for att in attributes:
        try:
            features.append(getattr(token, att + '_'))
        except:
            features.append(getattr(token, att))

    return features


def to_df(filename, delim, sep):
    sent_id = 0
    sent = []
    tags = []
    rows = []

    nlp = spacy.load('en_core_web_lg')
    nlp.tokenizer = lambda X: Doc(nlp.vocab, words=X)

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n' + sep)
            if line:
                row = line.split(delim)
                sent.append(row[0])
                tags.append(row[-1])
            else:
                doc = nlp(sent)
                for i, (token, tag) in enumerate(zip(doc, tags)):
                    rows.append(get_features(token, columns) +
                                [token.text, tag, sent_id, i])
                sent = []
                tags = []
                sent_id += 1

    df = pd.DataFrame.from_records(
        rows, columns=columns + ['token', 'tag', 'sentence_id', 'token_id'])

    return df


def json1_to_df(filename):
    '''for the test set which only has tag'''

    data = json_to_dict(filename)

    xy = []

    for _, datum in data.items():
        text = datum['text']
        labels = sorted(datum['labels'], key=lambda x: x[0])
        xi = [text[start:finish] for start, finish, _ in labels]
        yi = [tag for _, _, tag in labels]
        xy.append([xi, yi])

    df = pd.DataFrame.from_records(xy, columns=['tokens', 'labels'])
    return df


if __name__ == '__main__':
    '''
    type filename csvname
    '''

    assert len(sys.argv) >= 2, 'at least 1 argument accepted.'

    if 'json' in sys.argv[1]:
        df = json1_to_df(sys.argv[2])
    elif 'doccano_conll' in sys.argv[1]:
        df = to_df(sys.argv[2], '\t', '')
    elif 'conll' in sys.argv[1]:
        df = to_df(sys.argv[2], ' ', '')
    elif 'wnut' in sys.argv[1]:
        df = to_df(sys.argv[2], '\t', '\t')
    elif 'doccano' in sys.argv[1]:
        df = to_df(sys.argv[2], '\t', '')
    else:
        raise Exception(f'Unable to handle type {sys.argv[1]}')

    print(df.head(20))

    df.to_csv(sys.argv[3], index=False)
