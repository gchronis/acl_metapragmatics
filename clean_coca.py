"""
This script cleans up the useless censored portions of the COCA corpus.
COCA corpus has a special feature that removes ten tokens every 200 tokens and replaces them with an @ symbol. 
So we want to pass through the corpus and remove those sentences.

The resulting cleaned corpus is stored as a pandas dataframe in a parquet file for fast reading.

The data frame has the following format



Here is the basic overview of the logic of the script.

1. for each file
2. split the file on the doc id
3. keep the doc id
4. sentencize the doc with spacy
5. remove the sentences that have the @@@@@ symbols
6. (don't actually do this yet---add the genre)
7. put into a dataframe
8. write/append to a parquet file.

"""


import numpy as np
import pandas as pd
import os

#from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE
import pyarrow as pa
import pyarrow.parquet as pq

import csv
import re

import spacy
from collections import defaultdict 
from tqdm import tqdm



def metadata_frame_from_file(coca_path):
    columns = ["textID", "#words", "year", "genre", "subgen", "source", "title", "publication_info"]
    metadata = pd.read_csv(coca_path, skiprows=3, sep='\t', names = columns, encoding='unicode_escape', dtype={'genre': str, 'source': str, 'title': str, 'publication_info': str})


    """
    we need nice pkey values
    """
    
    # Convert the column to numeric, invalid parsing will be set as NaN
    metadata['textID'] = pd.to_numeric(metadata['textID'], errors='coerce')
    
    # Drop rows where 'textID' is NaN
    metadata = metadata.dropna(subset=['textID'])
    
    # Convert 'textID' back to integer type if desired
    metadata['textID'] = metadata['textID'].astype(np.int64)


    # Convert the column to numeric, invalid parsing will be set as NaN
    metadata['textID'] = pd.to_numeric(metadata['textID'], errors='coerce')
    metadata['#words'] = pd.to_numeric(metadata['#words'], errors='coerce')
    metadata['year'] = pd.to_numeric(metadata['year'], errors='coerce')
    #metadata['genre'] = pd.to_string(metadata['genre'], errors='coerce')
    metadata['subgen'] = pd.to_numeric(metadata['subgen'], errors='coerce')
    #metadata['source'] = pd.to_string(metadata['source'], errors='coerce')
    #metadata['publication_info'] = pd.to_string(metadata['publication_info'], errors='coerce')

    return metadata

def dataframe_from_file(path):
    
    # Read the entire content of the text file into a string
    with open(path) as file:
        text = file.read()
    
    # Define the regular expression pattern to match the delimiters
    pattern = r'@@\d{7}'
    
    # Find all matches of the pattern
    matches = list(re.finditer(pattern, text))
    print("number of docs in file: ", len(matches))
    
    # create a list of match texts, which are our document ids
    doc_ids = [match.group()[3:] for match in matches] # get rid of the initial special characters on str_id
    #print(len(doc_ids))
    
    # Initialize a list to store the captured segments, which are our document texts
    doc_texts = []
    
    # Add the text between matches
    start_idx = 0
    for match in matches:
        end_idx = match.start()
        if end_idx > start_idx:
            doc_texts.append(text[start_idx:end_idx].strip())
        start_idx = match.end()
    
    # Add the text after the last match
    if start_idx < len(text):
        doc_texts.append(text[start_idx:].strip())

    #print(len(doc_ids))
    #print(len(doc_texts))
    
    # Create a DataFrame from the captured segments
    df = pd.DataFrame(data = {'textID': doc_ids, 'doc_text': doc_texts})

    """
    we need nice pkey values for our join
    """
    
    # Convert the column to numeric, invalid parsing will be set as NaN
    df['textID'] = pd.to_numeric(df['textID'], errors='coerce')
    
    # Drop rows where 'textID' is NaN
    df = df.dropna(subset=['textID'])
    
    # Convert 'textID' back to integer type if desired
    df['textID'] = df['textID'].astype(np.int64)

    # filter out censored sentences
    cleaned_doctexts = []
    for text in df.doc_text:
        cleaned_doctext = strip_censored_sents(text)
        cleaned_doctexts.append(cleaned_doctext)
    df.doc_text = cleaned_doctexts

    return df


def strip_censored_sents(text):
    censor_string = "@ @ @ @ @ @ @ @ @ @"
    # Process the text with spaCy
    doc = nlp(text)
    
    # Initialize a list to store sentences containing the specific string
    filtered_sentences = []
    
    # Iterate over the sentences in the document
    for sent in doc.sents:
        #print(sent)
        if censor_string not in sent.text:
            filtered_sentences.append(sent.text.strip())

    return(' '.join(filtered_sentences))

def get_text_file_names(coca_path):
    # Define the root directory
    root_dir = coca_path + "texts"
    
    # List to store paths of all text files
    text_files = []
    
    # Walk through the directory
    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            sub_dir_path = os.path.join(root, dir_name)
            # Find all text files in the current subdirectory
            for txt_file in os.listdir(sub_dir_path):
                text_files.append(os.path.join(sub_dir_path, txt_file))
    return text_files


    
if __name__ == '__main__':


    _COCA_PATH = "/Volumes/data_gabriella_chronis/corpora/COCA/"
    _COCA_METADATA = "/Volumes/data_gabriella_chronis/corpora/COCA/shared_files/coca-sources.txt"
    _OUT_DIR = '/Volumes/data_gabriella_chronis/corpora/coca.2017.parquet'
    
    
    # initialize spacy nlp pipeline
    nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])
    nlp.add_pipe('sentencizer')

    # buid dataframe of metadata
    metadataframe = metadata_frame_from_file(_COCA_METADATA)

    # get all the fileneames for our corpus builder
    filenames = get_text_file_names(_COCA_PATH)

    #pqwriter = None

    # iterate through each file, reading in the contents and splitting it into a dataframe of filtered/cleaned documents.
    for i,path in tqdm(enumerate(filenames)):
        #print(i)
        print(path)
        
        df_one_file = dataframe_from_file(path)

        # join dataframe on id to add metadata
        merged_df = pd.merge(df_one_file, metadataframe, on='textID', how='left')


        schema = pa.schema([
            ('textID', pa.int64()),
            ('doc_text', pa.string()),
            ('#words', pa.int64()),
            ('year', pa.int64()),
            ('genre', pa.string()),
            ('subgen', pa.int64()),
            ('source', pa.string()),
            ('title', pa.string())
        ])
        
        table = pa.Table.from_pandas(merged_df, schema)
        # for the first chunk of records
        if i == 0:
            # create a parquet write object giving it an output file
            pqwriter = pq.ParquetWriter(_OUT_DIR, schema) #table.schema)            
        pqwriter.write_table(table)

        # old not working method
        #df_one_file.to_parquet(_OUT_DIR, mode='append')
        
    # close the parquet writer
    if pqwriter:
        pqwriter.close()
