# essentially this notebook 
# salient_cluster_features_model.ipynb

import preprocess_for_context_atlas
import torch
import pandas as pd
import numpy as np
from os import path
import pickle


sources = ["acl", "coca"]
#words = ["model"]
# words = ["chair", "bird", "football"]
words = ['no', 'first', 'one', 'third', 'large', 'high', 'clear', 'same', 'general', 'ready', 'age', 'information', 'word', 'door', 'meaning', 'government', 'study', 'animal', 'growth', 'building', 'left', 'seem', 'died', 'obtained', 'ran', 'built', 'considered', 'took', 'stand', 'suppose']
k_means_n = 5
model_name = 'roberta-base'
n_samples = 1000
layer = 7
#NOTE change the file names to match roberta for roberta-base or bert for bert-base

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device : ", device)

for word in words:
    print(word)
    for source in sources:
        print(source)

        # sample tokens of 'word' from source
        df = pd.read_csv('/home/gsc685/data/collected_tokens/{}/{}.csv'.format(source,word))
        df['source'] = source


        # filter sentences that are less than 150 in length
        df = df[ df["sentence"].apply(lambda x: len(x) < 300)]
        df["sentence"] = df["sentence"].apply(lambda x: x.encode('ascii', 'ignore').decode('ascii', 'ignore'))

        # Take at most n sentences.
        sample_df = df.sample(max(n_samples, len(df)), random_state=42) # use a fixed seed for reproducibility


        # load the (precalculated) features for this word and select the features for this sample
        feature_embs = np.loadtxt('/home/gsc685/data/collected_tokens/{}/{}_feature_vectors_roberta_buchanan_layer{}.txt'.format(source,word, layer))
        feature_sample = feature_embs[sample_df.index]
        
        processor = preprocess_for_context_atlas.Preprocessor(model_name=model_name, k=k_means_n, layer=layer)

        # calculate clusters for this sample
        filename = 'viz_data/{}/{}/{}.pkl'.format(model_name, source, word)
        if not path.exists(filename):
            # get the vectors
            data = processor.neighbors(word, sample_df)
            # save data
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
        else:
            with open(filename, 'rb') as f:
                data = pickle.load(f)


        sents = [d["sentence"] for d in data['labels'] ]
        
        clusters_df = pd.DataFrame({"cluster": data['clusters'][7], "sentence": sents}) # layer z
        
        # to see what the clustrs look like
        #clusters_df.groupby("cluster").sample(n=10)

        # get the names of the features
        buchanan_norms = pd.read_csv('/home/gsc685/semantic-features/feature-norms/buchanan/cue_feature_words.csv')
        name_col = 'translated'
        freq_col = 'frequency_'+name_col
        feature_labels = buchanan_norms[name_col].unique().tolist()



        ids = []
        sources = []
        sent = []
        cluster = []
        feature = []
        predicted_value = []
        for i, (index, row) in enumerate(sample_df.iterrows()):

            j = 0
            feature_vec = feature_sample[i]

            for value in feature_vec:
                #print(feature_labels[j])
                ids.append(row.token_id)
                sent.append(row.sentence)
                sources.append(row.source)
                cluster.append(clusters_df["cluster"].iloc[i])
                feature.append(feature_labels[j])
                predicted_value.append(value)
                j+=1
            j=0

        tidy_df = pd.DataFrame.from_records(
            {"sentence_id": ids,
            "source": sources,
            "word": word, 
            "cluster": cluster, 
            "feature": feature, 
            "predicted_value": predicted_value}
        )

        tidy_df.to_csv('./tidy_feature_predictions/{}/{}/{}_buchanan_layer_7.csv'.format(model_name,source,word))
