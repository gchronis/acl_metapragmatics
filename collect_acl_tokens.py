#from __future__ import print_function
#import time
#import numpy as np
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

_ACL_ARCHIVE_PATH = "/Volumes/data_gabriella_chronis/corpora/acl-publication-info.74k.parquet"
_OUT_DIR = './collected_tokens/acl'

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

def add_matches_to_dict(match_list, match_dict, doc_id, sent_text):
    """
    match list is in the form of a list of tuples [(match_phrase_id, start index, end index)]
    """
    for match in match_list:
        word = match[0]
        start_idx = match[1]
        end_idx = match[2]
        entry = {"corpus_id": doc_id, "sentence": sent_text, "start_idx": start_idx, "end_idx": end_idx}
        match_dict[word].append(entry)
    return match_dict

def get_matches_in_doc(doc_id, doc_text, phrase_matcher, matches=None):
    """
    searches through a docstring to extract a given token.
    ::
    """
    if matches is None:
        matches = defaultdict(list)
    
    doc = nlp(doc_text)
    for sent in doc.sents:
        # this will be a list of tuples
        this_sent_matches = phrase_matcher(nlp(sent.text))
        matches = add_matches_to_dict(this_sent_matches, matches, doc_id, sent)
        
    return matches


def get_matches_in_corpus(df, targets, phrase_matcher=None, outdir=None):

    if phrase_matcher is None:
        phrase_matcher = build_phrase_matcher(targets = targets)
    
    matches = defaultdict(list)
    for idx, row in tqdm(df.iterrows()):
        matches = get_matches_in_doc(row.corpus_paper_id, row.full_text, phrase_matcher, matches)
        #print(matches)
    #get the word as opposed to the spacy vocab id
    for key in list(matches.keys()):
        word = nlp.vocab.strings[key]
        
        matches_df = pd.DataFrame.from_records(matches.pop(key))
        if outdir:
            matches_df.to_csv(outdir+'/'+word+'.csv')

def load_acl_archive():
    parquet_file = _ACL_ARCHIVE_PATH
    return pd.read_parquet(parquet_file, engine='pyarrow')
    
if __name__ == '__main__':

    # relative path to desired output directory
    outdir = _OUT_DIR

    # Load the English spacy model without all the bells and whistles 
    # we literally only need the sentencizer or it takes a million years
    nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])
    nlp.add_pipe('sentencizer')

    # words we wish to collect sentences for
    #targets = ["toxic", "toxicity", "hallucination", "hallucinate", "safe", "safety"]
    targets = ["reference", "intention", "intension", "sense", "symbol", "symbolic", "index", "indexical", "icon", "iconic"]

    df = load_acl_archive()
    
    phrase_matcher = build_phrase_matcher(targets=targets)
    get_matches_in_corpus(df, targets, outdir=outdir, phrase_matcher=phrase_matcher)
    