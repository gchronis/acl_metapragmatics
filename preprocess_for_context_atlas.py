
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Edited by Gabriella Chronis 2024
#
# ==============================================================================

"""Preprocessing the data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
#from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from transformers import AutoTokenizer, AutoModel, AutoConfig
import sqlite3 as sql
import re
import numpy as np
import umap
import json
from tqdm import tqdm
import nltk
import pandas as pd

from sklearn.cluster import KMeans


DB_PATH = './enwiki-20170820.db'
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

## really shouldnt do this globally
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device : ", device)

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()
model = model.to(device)

def neighbors(word, df):
  """Get the info and (umap-projected) embeddings about a word."""
  # Get part of speech of this word.
  sentences  = df['sentence'].to_list()  
  sent_data = get_poses(word, sentences)

  # Get embeddings.
  points = get_embeddings(word.lower(), sentences)

  # Use UMAP to project down to 3 dimnsions.
  points_transformed = project_umap(points)

  clusters = cluster_embeddings(points, k=5)

  features = predict_features_for(points, model="buchanan")

  return {'labels': sent_data, 'data': points_transformed, 'clusters': clusters}

def project_umap(points):
  """Project the words (by layer) into 3 dimensions using umap."""
  points_transformed = []
  for layer in points:
    transformed = umap.UMAP().fit_transform(layer).tolist()
    points_transformed.append(transformed)
  return points_transformed

"""
GS chronis 03/24
"""
def get_embeddings(word, sentences):
  # always take the first occurrence of a word that appears twice in the sentnce
  word_occurrence = 0
    
  layers = range(-12, 0)
  points = [[] for layer in layers]
  print('Getting embeddings for %d sentences '%len(sentences))
  for sentence in sentences:
    inputs = tokenizer(sentence, truncation=True, return_tensors="pt")      
    words = [i[0]
        for i in tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(sentence)]
    target_word_indices = [i for i, x in enumerate(words) if x == word]
    encoded_text = model(
        **inputs, output_hidden_states=True)["hidden_states"]
    word_start, word_end = inputs.word_to_tokens(target_word_indices[word_occurrence])
    avg_vectors_for_target_word = torch.cat(encoded_text)[:,word_start:word_end,:].mean(dim=1)
    #print(avg_vectors_for_target_word.shape)

    # Reconfigure to have an array of layer: embeddings
    for l in layers:
      sentence_embedding = avg_vectors_for_target_word[l]
      sentence_embedding =  sentence_embedding.detach().numpy()
      points[l].append(sentence_embedding)
    
  points = np.asarray(points)
  return points 
    

def cluster_embeddings(points, k=5):
    """
    :points: an np.ndarray of bert embeddings of dimension [n_layers, n_words, n_dims] (i.e. [12,200,768])

    return: an 2D np.ndarray containing cluster ids of shape [n_layers, n_words]
    """
    num_layers = points.shape[0]
    clusters = []
    for l in range(0, num_layers):
        embs = points[l]
        
        
        kmeans_obj = KMeans(n_clusters=k)
        kmeans_obj.fit(embs)

        #label_list = kmeans_obj.labels_
        #cluster_centroids = kmeans_obj.cluster_centers_
        clusters.append( kmeans_obj.fit_predict(embs))

    return np.asarray(clusters)

def predict_features_for(points, model="buchanan"):
    """
    Not implemented yet!

    should return a matrix of feature predictions 
    """
    return None

def tokenize_sentences(text):
  """Simple tokenizer."""
  print('starting tokenization')

  text = re.sub('\n', ' ', text)
  sentences = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

  # Filter out too long sentences.
  sentences = [t for t in sentences if len(t) < 150]

  return sentences


def get_query(select, db=DB_PATH):
  """Executes a select statement and returns results and column/field names."""
  with sql.connect(db) as conn:
    c = conn.cursor()
    c.execute(select)
    col_names = [str(name[0]).lower() for name in c.description]
  return c.fetchall(), col_names


def get_sentences():
  """Returns a bunch of sentences from wikipedia"""
  print('Selecting sentences from wikipedia...')

  select = 'select * from articles limit 5000000'
  docs, _ = get_query(select)
  docs = [doc[3] for doc in docs]
  doc = ' '.join(docs)
  print('Number of articles selected: %d'%len(docs))

  sentences = tokenize_sentences(doc)
  print('Total number of sentences: %d'%len(sentences))
  np.random.shuffle(sentences)
  return sentences



def get_poses(word, sentences):
  """Get the part of speech tag for the given word in a list of sentences."""
  sent_data = []
  for sent in sentences:
    text = nltk.word_tokenize(sent)
    pos = nltk.pos_tag(text)
    try:
      word_idx = text.index(word)
      pos_tag = pos[word_idx][1]
    except:
      pos_tag = 'X'
    sent_data.append({
      'sentence': sent,
      'pos': pos_tag
    })

  return sent_data


if __name__ == '__main__':

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print("device : ", device)

  model_name = "bert-base-uncased"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModel.from_pretrained(model_name)
  model.eval()
  model = model.to(device)

  # # Load pre-trained model tokenizer (vocabulary)
  # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  # # Load pre-trained model (weights)
  # model = BertModel.from_pretrained('bert-base-uncased')
  # model.eval()
  # model = model.to(device)

  # Get selection of sentences from wikipedia.
  with open('static/words.json') as f:
    words = json.load(f)

  for word in tqdm(words):

    # load sentences
    sentences_w_word = pd.read_csv('./data/logic_words/{}.csv'.format(word))
    # filter out too long sentences
    sentences_w_word[
        sentences_w_word["sentence"].apply(lambda x: len(x) > 150)
    ]

    # Take at most 200 sentences.
    sentences_w_word = sentences_w_word.sample(200)

    # convert to list
    sentences_w_word = sentences_w_word['sentence'].to_list()



    # And don't show anything if there are less than 100 sentences.
    if (len(sentences_w_word) > 100):
      print('starting process for word : %s'%word)
      locs_and_data = neighbors(word, sentences_w_word)
      with open('static/jsons/%s.json'%word, 'w') as outfile:
        json.dump(locs_and_data, outfile)

  # Store an updated json with the filtered words.
  filtered_words = []
  for word in os.listdir('static/jsons'):
    word = word.split('.')[0]
    filtered_words.append(word)

  with open('static/filtered_words.json', 'w') as outfile:
    json.dump(filtered_words, outfile)
  print(filtered_words)