#from __future__ import print_function
#import time
#import numpy as np
import argparse
import pandas as pd

#from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE
import pyarrow
import fastparquet

import csv

import spacy
from collections import defaultdict 
from tqdm import tqdm

from spacy.matcher import PhraseMatcher, Matcher

def build_phrase_matcher(targets=None):
    """
    expects a dictionary of match labels and phrases. If not provided, uses targets defined in this function
    """
    print(targets)

    phrase_matcher = Matcher(nlp.vocab)
    # phrases = ['language', 'model', 'intelligence', 'predict', 'human']
    # patterns = [nlp(text) for text in phrases]
    # phrase_matcher.add('AI', None, *patterns)
    # phrase_matcher.add('say', None, *[nlp(text) for text in ['say', 'said', 'speak', 'spoke']])


    for word in targets:
        print(word)
        phrase_matcher.add(word, [[{"orth": word}]])
        nlp.vocab.strings.add(word)
    
    return phrase_matcher

def add_matches_to_dict(doc, match_list, match_dict, doc_id, token_id, sent_text):
    """
    match list is in the form of a list of tuples [(match_phrase_id, start index, end index)]
    """
    for match_id, start, end in match_list:
        word = match_id
        span = doc[start:end]
        assert len(span) == 1 # we should only have one word here
        pos = span[0].pos_
        entry = {"corpus_id": doc_id, 
                 "sentence_id": token_id, 
                 "sentence": sent_text,
                 "start_idx": span.start_char, 
                 "end_idx": span.end_char,
                 "pos": pos}
        match_dict[word].append(entry)
    return match_dict

def get_matches_in_doc(doc_id, doc_text, phrase_matcher, matches=None):
    """
    searches through a docstring to extract a given token.
    :: keeps passing match_dict and adding to it
    """
    if matches is None:
        matches = defaultdict(list)
    
    doc = nlp(doc_text)
    for i, sent in enumerate(doc.sents):
        token_id = i
        # this will be a list of tuples
        this_sent_matches = phrase_matcher(nlp(sent.text))
        matches = add_matches_to_dict(doc, this_sent_matches, matches, doc_id, token_id, sent)
    return matches


def get_matches_in_corpus(df, targets, phrase_matcher=None):

    if phrase_matcher is None:
        phrase_matcher = build_phrase_matcher(targets = targets)
    
    matches = defaultdict(list)
    for idx, row in tqdm(df.iterrows()):
        matches = get_matches_in_doc(row.textID, row.doc_text, phrase_matcher, matches)
        #print(matches)
    return matches

def load_archive(parquet_file):
    return pd.read_parquet(parquet_file, engine='pyarrow')
    
if __name__ == '__main__':

    _COCA_ARCHIVE_PATH = '/home/gsc685/data/coca.2017.parquet'
    _COCA_OUT_DIR = '/home/gsc685/data/collected_tokens/coca'
        # relative path to desired output directory
    outdir = _COCA_OUT_DIR

    parser = argparse.ArgumentParser(description="Parse a corpus file.")
    parser.add_argument('--corpus_path', type=str, required=False, default=_COCA_ARCHIVE_PATH, help="Path to the corpus file.")
    parser.add_argument('--out_dir', type=str, required=False, default=_COCA_OUT_DIR, help="Path to the corpus file.")
    args = parser.parse_args()

    # Load the English spacy model without all the bells and whistles 
    # we literally only need the sentencizer or it takes a million years
    nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])
    nlp.add_pipe('sentencizer')

    # words we wish to collect sentences for
    #targets = ["toxic", "toxicity", "hallucination", "hallucinate", "safe", "safety", "reference", "intention", "intension", "sense", "symbol", "symbolic", "index", "indexical", "icon", "iconic"]
    #targets = ["model", "models"]
    #targets = ["human", "harness", "helpful", "honest", "harmless", "friendly"]
    targets =  ['no', 'first', 'one', 'third', 'large', 'high', 'clear', 'same', 'general', 'ready', 'age', 'information', 'word', 'door', 'meaning', 'government', 'study', 'animal', 'growth', 'building', 'left', 'seem', 'died', 'obtained', 'ran', 'built', 'considered', 'took', 'stand', 'suppose']

    df = load_archive(args.corpus_path)
    
    phrase_matcher = build_phrase_matcher(targets=targets)
    # create a csv for every new word.
    matches_dict = get_matches_in_corpus(df, targets, phrase_matcher=phrase_matcher)
    
        #get the word as opposed to the spacy vocab id
    for key in list(matches_dict.keys()):
        word = nlp.vocab.strings[key] 
        matches_df = pd.DataFrame.from_records(matches_dict.pop(key))
        if outdir:
            matches_df.to_csv(outdir+'/'+word+'.csv')
