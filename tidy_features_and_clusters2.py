import argparse
import torch
import pandas as pd
import numpy as np
from os import path, makedirs
import pickle
import preprocess_for_context_atlas

def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    return device

def load_data(source, word, layer):
    df = pd.read_csv(f'/home/gsc685/data/collected_tokens/{source}/{word}.csv')
    df['source'] = source
    feature_embs = np.loadtxt(f'/home/gsc685/data/collected_tokens/{source}/{word}_feature_vectors_roberta_buchanan_layer{layer}.csv')
    
    # need to add "sentence" column to feature_embs for embedding all over again for clusterizing. 
    indexes = features_embs['sentence_id'].tolist()
    sentences = df.iloc[indexes]
    feature_embs['sentence'] = sentences

    return feature_embs


def load_or_generate_clusters(processor, word, sample_df, model_name, source):
    filename = f'/home/data/viz_data/{model_name}/{source}/{word}.pkl'
    if not path.exists(filename):
        makedirs(path.dirname(filename), exist_ok=True)
        data = processor.neighbors(word, sample_df)
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    else:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
    return data


def build_tidy_df(sample_df, feature_sample, clusters_df, feature_labels, word):
    records = []
    for i, (index, row) in enumerate(sample_df.iterrows()):
        for j, value in enumerate(feature_sample[i]):
            records.append({
                "sentence_id": row.token_id,
                "source": row.source,
                "word": word,
                "cluster": clusters_df["cluster"].iloc[i],
                "feature": feature_labels[j],
                "predicted_value": value
            })
    return pd.DataFrame.from_records(records)


def process_word_source(processor, word, source, model_name, k_means_n, layer):
    feature_sample = load_data(source, word, layer)
    cluster_data = load_or_generate_clusters(processor, word, feature_sample, model_name, source)
    sents = [d["sentence"] for d in cluster_data['labels']]
    clusters_df = pd.DataFrame({"cluster": cluster_data['clusters'][layer], "sentence": sents})

    buchanan_norms = pd.read_csv('/home/gsc685/semantic-features/feature-norms/buchanan/cue_feature_words.csv')
    feature_labels = buchanan_norms['translated'].unique().tolist()

    tidy_df = build_tidy_df(sample_df, feature_sample, clusters_df, feature_labels, word)

    output_dir = f'./tidy_feature_predictions/{model_name}/{source}'
    makedirs(output_dir, exist_ok=True)
    tidy_df.to_csv(f'{output_dir}/{word}_buchanan_layer_{layer}.csv', index=False)


def main():
    #sources = ["acl", "coca"]
    sources = ["semcor"]
    # words = ["model"]
    # words = ["chair", "bird", "football"]
    words = ['no', 'first', 'one', 'third', 'large', 'high', 'clear', 'same', 'general', 'ready', 'age', 'information', 'word', 'door', 'meaning', 'government', 'study', 'animal', 'growth', 'building', 'left', 'seem', 'died', 'obtained', 'ran', 'built', 'considered', 'took', 'stand', 'suppose']
    kmeans_n = 5
    model_name = 'roberta-base'
    layer = 7

    parser = argparse.ArgumentParser(description="Process feature predictions for specified words.")
    parser.add_argument('--words', nargs='+', default=words, help='List of words to process')
    parser.add_argument('--sources', nargs='+', default=sources, help='List of data sources')
    parser.add_argument('--model_name', default=model_name, help='Model name')
    parser.add_argument('--k_means_n', type=int, default=kmeans_n, help='Number of k-means clusters')
    parser.add_argument('--layer', type=int, default=layer, help='Model layer to use')
    args = parser.parse_args()

    get_device()

    for word in args.words:
        print(f"Processing word: {word}")
        for source in args.sources:
            print(f"\tSource: {source}")
            processor = preprocess_for_context_atlas.Preprocessor(model_name=model_name, k=k_means_n, layer=layer)
            process_word_source(processor, word, source, args.model_name, args.k_means_n, args.layer)

if __name__ == "__main__":
    main()
