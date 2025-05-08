from nltk.corpus import semcor, brown, wordnet as wn
from nltk.corpus.reader.wordnet import Synset
import random
import pandas as pd
from nltk.corpus.reader.wordnet import Lemma
from nltk.tree import Tree
from nltk.stem import WordNetLemmatizer
import nltk
from tqdm import tqdm



# Function to extract the corresponding Brown ID from a SemCor file ID
def semcor_to_brown_fileid(semcor_id):
    # Example: 'br-a01.xml' → 'a01'
    return semcor_id[19:22]

def get_brown_mapping():
    """
    Get mapping from Brown file ID to category
    """
    # Step 1: Get mapping from Brown file ID to category
    brown_fileid_to_category = {fid: brown.categories(fid)[0] for fid in brown.fileids()}

    # Step 2: Get SemCor file IDs (these are like 'brown1/tagfiles/br-a01.xml')
    semcor_fileids = semcor.fileids()
    # get the brown category for the semcor sense


    # Step 3: Build mapping: SemCor file ID → Brown category
    semcor_file_to_category = {}
    for fid in semcor_fileids:
        brown_id = semcor_to_brown_fileid(fid)
        brown_fileid = next((x for x in brown.fileids() if brown_id in x), None)
        if brown_fileid:
            semcor_file_to_category[fid] = brown.categories(brown_fileid)[0]
    return semcor_file_to_category


def get_senses_in_tagged_sentence(tagged_sentence, lemmatizer, fileid, category, sent):
    """
    given a sense-tagged corpus sentence,returns a list of lemmas and senses in that sentence
    """
    res = []

    for chunk in tagged_sentence:
        if  isinstance(chunk, Tree) and isinstance(chunk.label() , Lemma):
            """
            if we find a wordnet sense (function words dont)
            then scoop it up

            """    

            chunk_string = ' '.join(chunk.leaves())
            word = chunk_string.lower()
            lemma = lemmatizer.lemmatize(word)
            poss = chunk.pos()
            sense = chunk.label()

            for wordform, pos in poss:
                row = {
                    'lemma': lemma,
                    'sense': str(sense),
                    'word_form': wordform.lower(),
                    'pos': pos,
                    'sentence': sent,
                    'category': category,
                    'fileid': fileid,
                    'domain': sense.synset().lexname()
                }
                res.append(row)
    return res

def collect_tokens(lemmatizer, semcor_file_to_category):
    """
    Next step is to create an index of all of the tokens of a single lemma. 
    So, we build a data structure with all of the word forms found in semcor. With each word form,
    we store a list of all of the sentences containing it.

    returns a df with columns
        lemma wn_sense word_form token_id
    """
    tokens = []

    for fileid in semcor.fileids():

        category = semcor_file_to_category.get(fileid, 'unknown')
        tagged_sentences = semcor.tagged_sents(fileids=[fileid], tag='sense')
        sentences = semcor.sents(fileids=[fileid])


        # go through the dataset sentence by sentence
        for sent, tagged_sent in tqdm(zip(sentences, tagged_sentences)):
            # go through the sentence word by word to get semcor senses in it
            sent = ' '.join(sent)
            senses = get_senses_in_tagged_sentence(tagged_sent, lemmatizer, fileid, category, sent)
            tokens += senses
    return pd.DataFrame.from_records(tokens)
    
if __name__ == '__main__':
    # Ensure required corpora are downloaded
    nltk.download('semcor')
    nltk.download('brown')

    semcor_file_to_category = get_brown_mapping()
    lemmatizer = WordNetLemmatizer()
    tokens_df = collect_tokens(lemmatizer, semcor_file_to_category)

    tokens_df.to_csv('/home/gsc685/data/semcor_all_tokens.csv', index=True, index_label='id')


    corpus_df = pd.DataFrame.from_dict({
        'sentence': [' '.join(ts) for ts in semcor.sents()],
    })

    corpus_df.to_csv('/home/gsc685/data/semcor_corpus.csv', index=True, index_label='id')

    tokens_df['lemma'] = tokens_df['lemma'].str.strip()
    tokens_df['word_form'] = tokens_df['word_form'].str.strip()

    # save each word in a separate file
    for word in tokens_df['lemma'].unique():
        word_df = tokens_df[tokens_df['lemma'] == word]
        word_df.to_csv(f'/home/gsc685/data/collected_tokens/semcor/{word}.csv', index=True, index_label='id')