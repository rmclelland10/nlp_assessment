#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ross
"""

import re
import pandas as pd
from tabulate import tabulate

def get_raw_text(filename):
    with open(filename, 'r') as file:
        text = file.read().replace('\n', ' ')
        text = text.replace("'", '')
        text = text.lower()
    return text

def sorted_display(text):
    percentages_and_results = re.findall('[\d]+%|\([+-]\)', text)
    
    for item in set(percentages_and_results):
        text = text.replace(item, ' ')
        
    words = re.findall('[a-z0-9\-]+[\-]?[a-z0-9]+|[a-z0-9]', text)
    tokens =pd.DataFrame(percentages_and_results + words, columns= ['Word'])
    sorted_freq = tokens.groupby('Word').size().sort_values(ascending=False).to_frame('Count').reset_index()
    result_display = tabulate(sorted_freq, headers='keys', tablefmt='psql', showindex=False)
    return result_display, tokens
    
def get_word_freq(text, search_word=None):
    tokens= sorted_display(text=text)[1]
    count = len(tokens[tokens['Word'] == search_word])
    search_result = tabulate([[search_word, count]], headers=['Word', 'Count'], tablefmt='psql')
    return search_result

if __name__ == '__main__':
    '''
    HOW TO RUN:
        ./oa_p1.py -task=1  ---> Runs sorted display on exercise_text.txt
        ./oa_p1.py -file=sample.txt -task=1  ---> Runs sorted display on a different file
        ./oa_p1.py -task=2 -search=patient  ---> Displays count of 'patient' occurences in exercise_text.txt
        ./oa_p1.py -file=sample.txt -task=2 -search=patient  ---> Displays count of 'patient' occurences in exercise_text.txt
    '''
    
    import argparse
    p = argparse.ArgumentParser(description='Oncology Analytics Assessment - Part 1.')
    p.add_argument('-file', type=str, help='Text file to search', default='exercise_text.txt')
    p.add_argument('-task', type=int, help='1 --> Sorted Display, 2 --> Return count for individual word')
    p.add_argument('-search', type=str, help='Search word to return count for')
    args = p.parse_args()
    
    if args.task == 2 and args.search is None:
        p.error('-task=2 requires -search to be set.')
  
    raw_text = get_raw_text(args.file)
    
    if args.task == 1:
        print(sorted_display(text=raw_text)[0])
    elif args.task == 2:
        print(get_word_freq(text=raw_text, search_word=args.search.lower()))
