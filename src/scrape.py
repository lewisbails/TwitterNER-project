import tweepy
import preprocessor as p
# import pandas as pd
from datetime import datetime as dt
import sys
import re

# p.set_options(p.OPT.URL,p.OPT.EMOJI,p.OPT.SMILEY)


def get_auth(key_path):
    with open(key_path,'r') as f:
        consumer_key = f.readline().rstrip()
        consumer_secret = f.readline().rstrip()
        access_key = f.readline().rstrip()
        access_secret = f.readline().rstrip()

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)

    return auth

def get_api(auth):
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    return api

def process(text):
    processed_text = re.subn(r'\s\s+',' ',text)[0]
    processed_text = p.clean(processed_text)
    processed_text = processed_text
    return processed_text

class Listener(tweepy.StreamListener):
    def __init__(self, output_file=sys.stdout):
        super(Listener,self).__init__()
        self.output_file = output_file

    def on_status(self, status):
        if not status.retweeted and 'RT @' not in status.text:
            try:
                text = status.full_text
            except:
                text = status.text
            tweet = process(text)
            print(tweet,status.truncated)
            print(tweet, file=self.output_file)

if __name__=='__main__':
    auth = get_auth('../keys/keys.txt')
    api = get_api(auth)
    output = open('../data/'+dt.now().strftime("%d_%m_%y")+'.txt','w')
    stream = tweepy.Stream(auth=api.auth, listener=Listener(output))

    try:
        print('Start streaming... ', end='')
        stream.sample(languages=['en'])
    except KeyboardInterrupt as e:
        print('Stopped!')
    finally:
        print('Done!')
        stream.disconnect()
        output.close()
