import tweepy
import preprocessor as p
# import pandas as pd
from datetime import datetime as dt
import sys
import re
import time
import os
from io import BufferedWriter

p.set_options(p.OPT.SMILEY,p.OPT.MENTION,p.OPT.URL)


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
    processed_text = text.replace("&amp;", "and").replace("&gt;", ">").replace("&lt;", "<").replace('…','').strip(' :-*^,+|_')
    
    # we currently remove the mention but we could also swap it out for a common handle
    # mentionFinder = re.compile(r"@\w{1,15}", re.IGNORECASE)
    # processed_text = mentionFinder.subn("@mention", processed_text)[0]

    processed_text = re.subn(r'\s\s+',' ',processed_text)[0]
    processed_text = p.clean(processed_text)
    processed_text += '\n'
    processed_text = processed_text.encode('utf-8')
    return processed_text

def write_dict(output_dict,filename):
    mode = 'ab' if os.path.exists(filename) else 'wb'
    with open(filename,mode) as f:
        for v in output_dict.values():
            f.write(v)

def end(stream,output):
    stream.disconnect()
    try:
        name = output.name
        output.close()
    except:
        name = '../data/'+dt.now().strftime("%d_%m_%y")+'.txt'
        write_dict(output,name)
    print(f'Saved to {name}.')


class Listener(tweepy.StreamListener):
    def __init__(self, output=sys.stdout):
        super(Listener,self).__init__()
        self.output = output

    def on_status(self, status):
        if not status.retweeted and 'RT @' not in status.text and not status.truncated and len(status.text)>50 and 'automatically checked' not in status.text :
            try:
                text = status.full_text
            except:
                text = status.text
            tweet = process(text)
            if len(tweet)>50:
                if isinstance(self.output,BufferedWriter):
                    self.output.write(tweet)
                else:
                    self.output[status.id] = tweet

if __name__=='__main__':
    auth = get_auth('../keys/keys.txt')
    api = get_api(auth)

    # write to file, not dictionary
    # filename = '../data/'+dt.now().strftime("%d_%m_%y")+'.txt'
    # mode = 'ab' if os.path.exists(filename) else 'wb'
    # output = open(filename,mode)

    output = dict()

    iters = sys.argv[1] if len(sys.argv)==2 else 5

    # using filter, not stream
    # will have to multiple connections as Stream.filter just returns 1 result set
    for _ in range(iters):
        try:
            listener = Listener(output)
            stream = tweepy.Stream(auth=api.auth, listener=listener)
            print('Start streaming... ', end='')
            stream.filter(track=['i','you','me','us','they','we','he','she','the','it','this','that'],languages=['en'], is_async=True)
            time.sleep(3)
            raise Exception
        except:
            print('Stopped!')
        finally:
            stream.disconnect()

    end(stream,listener.output)

    # constant streaming
    # try:
    #     print('Start streaming... ', end='')
    #     stream.sample(languages=['en'], is_async=True)
    #     try:
    #         time.sleep(sys.argv[1])
    #     except:
    #         time.sleep(5)
    #     raise KeyboardInterrupt
    # except KeyboardInterrupt:
    #     print('Stopped! ', end='')
    # finally:
    #     end(stream,listener.output)
