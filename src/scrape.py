import tweepy
import preprocessor as p
# import pandas as pd
from datetime import datetime as dt
import sys
import re
import time
import os
import json
from io import BufferedWriter
import configparser


def get_auth(key_path):
    with open(key_path, 'r') as f:
        consumer_key = f.readline().rstrip()
        consumer_secret = f.readline().rstrip()
        access_key = f.readline().rstrip()
        access_secret = f.readline().rstrip()

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)

    return auth


def get_api(auth):
    api = tweepy.API(auth, wait_on_rate_limit=True,
                     wait_on_rate_limit_notify=True)
    return api


def process(text):
    processed_text = text.replace("&amp;", "and").replace(
        "&gt;", ">").replace("&lt;", "<").replace('…', '').strip(' :-*^,+|_')

    # we currently remove the mention but we could also swap it out for a common handle
    # mentionFinder = re.compile(r"@\w{1,15}", re.IGNORECASE)
    # processed_text = mentionFinder.subn("@mention", processed_text)[0]

    processed_text = re.subn(r'\s\s+', ' ', processed_text)[0]
    processed_text = p.clean(processed_text)
    processed_text += '\n'
    processed_text = processed_text.encode('utf-8')
    return processed_text


def write_dict(output_dict, filename):
    mode = 'ab' if os.path.exists(filename) else 'wb'
    with open(filename, mode) as f:
        for v in output_dict.values():
            f.write(v)


class Listener(tweepy.StreamListener):
    def __init__(self, output=sys.stdout):
        super(Listener, self).__init__()
        self.output = output

    def on_status(self, status):
        if not status.retweeted and not status.truncated and len(status.text) > 50 and not any(x in status.text for x in ['RT @', 'automatically checked', '... More for ']):
            try:
                text = status.full_text
            except:
                text = status.text
            tweet = process(text)
            if len(tweet) > 50:
                if isinstance(self.output, BufferedWriter):
                    self.output.write(tweet)
                else:
                    self.output[status.id] = tweet

    def on_end(self):
        try:
            name = output.name
            self.output.close()
        except:
            name = '../data/' + dt.now().strftime("%d_%m_%y") + '.txt'
            write_dict(self.output, name)
        print(f'Saved to {name}.')


if __name__ == '__main__':
    # configuration
    config = configparser.ConfigParser()
    config.read('config.ini')
    config = config[sys.argv[1]]

    # set options on cleaning
    options = [eval(f'p.OPT.{option}')
               for option in config['options'].split(',')]
    p.set_options(*options)

    # set API keys
    auth = get_auth(config['keys'])
    API = get_api(auth)

    # setup output file
    filename = config['output'] + dt.now().strftime("%d_%m_%y") + '.txt'
    if config['to_dict']:
        mode = 'ab' if os.path.exists(filename) else 'wb'
        output = open(filename, mode)
    else:
        output = dict()

    # set seeds
    seeds = {}
    try:
        with open(config['seeds'], 'r') as f:
            for line in f:
                entity_seeds = line.rstrip('\n').split(' ')
                seeds[entity_seeds[0]] = entity_seeds[1:]
    except:
        pass

    # start streaming tweets
    listener = Listener(output)

    if eval(config['filter']) and len(seeds):
        for e_type, track in seeds.items():
            try:
                print(f'Start streaming for {e_type}... ', end='')
                for status in tweepy.Cursor(API.search, q=' OR '.join(track), count=1000, lang='en').items(100):
                    listener.on_status(status)
                raise Exception
            except:
                print('Stopped!')
    else:
        try:
            print('Start streaming... ', end='')
            for status in tweepy.Cursor(API.search, q='*', count=100, lang='en').items(10000):
                listener.on_status(status)
            raise Exception
        except Exception:
            print('Stopped! ', end='')

    listener.on_end()
