from nltk.corpus import semcor
import random
import pandas as pd
from nltk.corpus.reader.wordnet import Lemma
from nltk.stem import WordNetLemmatizer
import nltk
from tqdm import tqdm




def load_corpus():
    """
    load semcor stats
    """
    nltk.download('semcor')
    nltk.download('wordnet')

    #uncomment for whole dataset
    sents = semcor.sents()
    tagged_sents = semcor.tagged_sents( tag = ' sem ' )
    words = semcor.words()
    return sents, tagged_sents, words


def get_senses_in_tagged_sentence(tagged_sentence, lemmatizer):
    """
    given a sense-tagged corpus sentence,returns a list of lemmas and senses in that sentence
    
    """
    res = []
    for chunk in tagged_sentence:
        
        chunk_string = ' '.join(chunk.leaves())

        word = chunk_string.lower()
        lemma = lemmatizer.lemmatize(word)
        poss = chunk.pos()
        
        """
        if we find a wordnet sense (function words dont)
        then scoop it up

        """            
        if isinstance(chunk.label() , Lemma):
            sense = chunk.label()
            for wordform, pos in poss:
                res.append((lemma, sense, wordform.lower(), pos))
    return res

def collect_tokens(sents, tagged_sents, words, lemmatizer):
    """
    Next step is to create an index of all of the tokens of a single lemma. 
    So, we build a data structure with all of the word forms found in semcor. With each word form,
    we store a list of all of the sentences containing it.

    returns a df with columns
        lemma wn_sense word_form token_id
    """
    tokens = []

    semcor_indices = list(range(0,len(tagged_sents)))
    random.shuffle(semcor_indices)

    # go through the dataset sentence by sentence
    for random_index in tqdm(semcor_indices):

        sentence_id = random_index
        sent = tagged_sents[sentence_id]

        
        # go through the sentence word by word to get semcor senses in it
        senses = get_senses_in_tagged_sentence(sent, lemmatizer)
        for lemma, sense, wordform, pos in senses:                
            row = {
                'lemma': lemma,
                'sense': str(sense),
                'word_form': wordform,
                'sentence_id': sentence_id, 
                'pos': pos
            }
            tokens.append(row)
    return pd.DataFrame.from_records(tokens)
    
if __name__ == '__main__':
    sents, tagged_sents, words = load_corpus()
    lemmatizer = WordNetLemmatizer()
    tokens_df = collect_tokens(sents, tagged_sents, words, lemmatizer)

    tokens_df.to_csv('/home/gsc685/data/semcor_all_tokens.csv', index=False)

    corpus_df = pd.DataFrame.from_dict({
        'sentence': [' '.join(ts) for ts in sents],
        'id': range(0,len(sents))
    })
    corpus_df.to_csv('/home/gsc685/data/semcor_corpus.csv', index=False)