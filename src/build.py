import json
import pandas as pd
import sys
import spacy
from spacy.tokens import Doc

''' For turning CONLL-2003 data to a CSVs with a multitude of columms'''


columns = ['pos', 'dep', 'lemma', 'norm', 'lower', 'shape', 'is_alpha', 'is_ascii', 'is_digit', 'is_lower',
           'is_upper', 'is_title', 'is_punct', 'like_num', 'is_oov', 'is_stop', 'cluster', 'like_url', 'is_currency']


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


if __name__ == '__main__':
    '''
    sys.argv[1:] : filename csvname delimiter separator

    Turns conll format to csv
    '''

    filename = sys.argv[1]
    output = sys.argv[2]
    delim = sys.argv[3]
    sep = sys.argv[4]

    df = to_df(filename, delim, sep)

    print(df.head(20))

    df.to_csv(output, index=False)
