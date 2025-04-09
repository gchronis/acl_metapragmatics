
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
import umap.umap_ as umap
import json
from tqdm import tqdm
import nltk
import pandas as pd
from minicons import cwe
from torch.utils.data import DataLoader
import spacy


from sklearn.cluster import KMeans


DB_PATH = './enwiki-20170820.db'
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')





class Preprocessor():
    
    def __init__(self, model_name='bert-base-uncased', k=5, device="cuda:1"):
        ## really shouldnt do this globally
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print("device : ", self.device)

        self.model_name = model_name
        self.k = k
        self.embedding_model = cwe.CWE(model_name, device = device)
        self.nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer"])

    def neighbors(self, word, df):
      """Get the info and (umap-projected) embeddings about a word."""
      sentences = df['sentence'].to_list()
      df = df.reset_index(drop=True)
      # Get embeddings.
      good_indices, points = self.get_embeddings(word.lower(), sentences)

      # Get part of speech of this word.
      # filter sentences we couldnt get embeddings for
      print("successfully embedded sentences: ", len(good_indices))
      print("num embeddings: ", len(points[0]))
      print(len(df))
      print(df.head)
      df =  df.loc[df.index.isin(good_indices)]
      print(len(df))
      sent_data = get_poses(word, df)


      # Use UMAP to project down to 3 dimnsions.
      points_transformed = self.project_umap(points)

      clusters = self.cluster_embeddings(points)

      return {'labels': sent_data, 'data': points_transformed, 'clusters': clusters}

    def project_umap(self, points):
      """Project the words (by layer) into 3 dimensions using umap."""
      points_transformed = []
      for layer in points:
        transformed = umap.UMAP().fit_transform(layer).tolist()
        points_transformed.append(transformed)
      return points_transformed

    """
    GS chronis 03/24
    """
    def get_embeddings(self, word, sentences):
      # empty array for embeddings




      # Get hidden size (embedding & hidden layer size)
      num_dims = self.embedding_model.model.config.hidden_size
      
      # set batch size
      batch_size=75


      # data as list of tuples
      # data = list(zip(sentences, word))

      # save all queries separately
      # (needed because some words do not occur in
      # sentences in the same form and must be fixed first)
      queries = []
      for sentence in sentences:

          # get the word's span
          wordspan = self._find_word_form(word, sentence)
          # kick out sentences that are too long for the model
          # if len(sentence) <= self.embedding_model.tokenizer.model_max_length:
          #     queries.append((sentence, torch.tensor(wordspan)))
          queries.append((sentence, torch.tensor(wordspan)))

      embeddings = []
      nans = []
      for i, batch in tqdm(enumerate(batch_iterable(queries, batch_size))):

        embs = self.embedding_model.extract_representation(batch, layer='all')

        # just look at the first layer bc it will be the same for all
        layer1 = embs[0]

        # Check for NaN values
        nan_mask = torch.isnan(layer1)

        # Print the locations where NaNs are present
        nan_locations = torch.nonzero(nan_mask)
        # print(nan_locations.shape)
        rows_with_nan = torch.any(nan_mask, dim=1).nonzero(as_tuple=True)[0].numpy()
        #print("rows with nan")
        #print(rows_with_nan)

        # for k in rows_with_nan:
        #     print("can't get emb for ")
        #     print(batch[k])

        nan_indices = [row + (batch_size*i) for row in rows_with_nan] # get indices of problem data
        nans += nan_indices

        # Convert to NumPy array
        numpy_embs = np.array([layer_emb.detach().cpu().numpy() for layer_emb in embs]) # yelds |13 X 1000 X 768
        #print(numpy_embs.shape)
        numpy_embs = np.delete(numpy_embs, rows_with_nan, axis=1) # remove nan rows from this batch
        embeddings.append(numpy_embs)

      print("couldnt get embs for data at indices", nans )

      # put batches together
      embeddings = np.concatenate(embeddings, axis=1) # dimension 1 is num_sentences i.e. batch size and the dimension we want to concatenate on 
      print(embeddings.shape)

      good_indices = np.delete( np.arange(len(sentences)), nans, axis =0)
      print(len(good_indices))
      print(len(embeddings))
      return good_indices, embeddings


    def cluster_embeddings(self, points):
        """
        :points: an np.ndarray of bert embeddings of dimension [n_layers, n_words, n_dims] (e.g. [12,200,768])

        return: an 2D np.ndarray containing cluster ids of shape [n_layers, n_words]
        """
        num_layers = points.shape[0]
        clusters = []
        for l in range(0, num_layers):
            embs = points[l]

            #arr_cleaned = embs[~np.isnan(embs).any(axis=1)]
            kmeans_obj = KMeans(n_clusters=self.k, n_init=10)
            kmeans_obj.fit(embs)

            #label_list = kmeans_obj.labels_
            #cluster_centroids = kmeans_obj.cluster_centers_
            preds = kmeans_obj.fit_predict(embs)

            # preds = []
            # for emb in embs:
            #   try:
            #     prediction = kmeans_obj.fit_predict(emb)
            #     preds.append(prediction)
            #   except:
            #     preds.append(None)
            
            clusters.append( preds)

        return np.asarray(clusters)

    def predict_features_for(self, points, model="buchanan"):
        """
        Not implemented yet!

        should return a matrix of feature predictions 
        """
        return None
    
    def _find_word_form(self, word, sentence):
      """find how word occurs in sentence"""
      doc = self.nlp(sentence)
      for token in doc:
          if token.text == word:
              return (token.idx, token.idx + len(token.text))
          
      raise Exception("target token {} not in sentence {}".format(word, sentence))

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



def get_poses(word, df):
  """Get the part of speech tag for the given word in a list of sentences."""
  #sentences = df['sentence'].to_list()

  sent_data = []
  for index, row in df.iterrows():
    text = nltk.word_tokenize(row.sentence)
    pos = nltk.pos_tag(text)
    try:
      word_idx = text.index(word)
      pos_tag = pos[word_idx][1]
    except:
      pos_tag = 'X'
    sent_data.append({
      'sentence': row.sentence,
      'pos': pos_tag,
      'source': row.source
    })

  return sent_data


# helper function to batch process inputs
def batch_iterable(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


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
