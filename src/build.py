import json
import pandas as pd
import sys

def load(filename):
    with open(filename,'r',encoding='utf-8') as f:
        sents = {i:json.loads(line) for i,line in enumerate(f)}
    
    return sents

if __name__=='__main__':

    assert len(sys.argv) == 2, '1 argument accepted.'

    data = load(sys.argv[1])
    xy = []

    for row,datum in data.items():
        text = datum['text']
        labels = sorted(datum['labels'],key=lambda x: x[0])
        xi = [text[a:b] for a,b,_ in labels]
        yi = [c for _,_,c in labels]
        xy.append([xi,yi])
    
    df = pd.DataFrame.from_records(xy,columns=['tokens','labels'])

    print(df.head())

    df.to_csv('../data/train.csv')