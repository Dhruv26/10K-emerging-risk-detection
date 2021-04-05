import os
import pickle
from collections import defaultdict, Counter
from glob import glob
from itertools import chain, groupby
from typing import Dict, List, Tuple

import fuzzywuzzy
from fuzzywuzzy import process
import numpy as np
from sentence_transformers import SentenceTransformer
import sentence_transformers.util
from sklearn.cluster import AgglomerativeClustering

from config import Config
from risk_detection.utils import (create_dir_if_not_exists,
                                  get_file_name_without_ext, window)

embedder = SentenceTransformer('stsb-roberta-base')


def cluster(corpus: List[str]) -> Tuple[Dict[str, int], Dict[int, List[str]]]:
    corpus_embeddings = embedder.encode(corpus)
    # Normalize embeddings to unit length
    corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings,
                                                           axis=1,
                                                           keepdims=True)

    clustering_model = AgglomerativeClustering(n_clusters=None,
                                               distance_threshold=1.5)
    clustering_model.fit(corpus_embeddings)

    clustered_sentences = defaultdict(list)
    labels = clustering_model.labels_
    for sentence_id, cluster_id in enumerate(labels):
        clustered_sentences[cluster_id].append(corpus[sentence_id])

    return dict(zip(corpus, labels)), clustered_sentences


def find_cluster_matches(word_prev_cluster, prev_clusters, word_cluster,
                         curr_clusters):
    matches = dict()

    for cluster_num, cluster_words in curr_clusters.items():
        prev_cluster_nums = list()
        for word in cluster_words:
            words = process.extract(word, word_prev_cluster.keys(), limit=5)
            prev_cluster_nums.extend([word_prev_cluster[w[0]] for w in words])

        counts = Counter(prev_cluster_nums)
        # Check if we have a perfect match
        for prev_cluster_num, _ in counts.most_common():
            if set(cluster_words) == set(prev_clusters[prev_cluster_num]):
                matches[cluster_num] = prev_cluster_num
                continue

        prev_closest_cluster_num = counts.most_common()[0][0]
        common_words = set(cluster_words).intersection(
            set(prev_clusters[prev_closest_cluster_num]))
        if len(common_words) >= (min(len(cluster_words), len(
                prev_clusters[prev_closest_cluster_num])) // 2):
            matches[cluster_num] = prev_closest_cluster_num

    return matches


def find_cluster_matches_semantic(word_prev_cluster, prev_clusters,
                                  word_cluster, curr_clusters):
    matches = dict()

    prev_corpus = list(word_prev_cluster.keys())
    curr_corpus = list(word_cluster.keys())
    curr_corpus_lookup = {k: v for v, k in enumerate(curr_corpus)}

    prev_emb = embedder.encode(prev_corpus)
    curr_emb = embedder.encode(curr_corpus)
    semantic_matches = sentence_transformers.util.semantic_search(
        curr_emb, prev_emb, top_k=5
    )

    for cluster_num, cluster_words in curr_clusters.items():
        prev_cluster_nums = list()
        for word in cluster_words:
            matches_word_indices = semantic_matches[curr_corpus_lookup[word]]
            prev_cluster_nums.extend(
                [word_prev_cluster[prev_corpus[w['corpus_id']]] for w in
                 matches_word_indices])

        counts = Counter(prev_cluster_nums)
        # Check if we have a perfect match
        for prev_cluster_num, _ in counts.most_common():
            if set(cluster_words) == set(prev_clusters[prev_cluster_num]):
                matches[cluster_num] = prev_cluster_num
                continue

        prev_closest_cluster_num = counts.most_common()[0][0]
        common_words = set(cluster_words).intersection(
            set(prev_clusters[prev_closest_cluster_num]))
        if len(common_words) >= (
                min(len(cluster_words),
                    len(prev_clusters[prev_closest_cluster_num])
                    ) // 4
        ):
            matches[cluster_num] = prev_closest_cluster_num

    return matches


def write_yearly_keyword_clusters():
    import risk_detection.analysis.keyword_extraction
    keywords = risk_detection.analysis.keyword_extraction.get_all_keywords()

    print('Creating keyword clusters by year.')
    keyword_clusters_by_year = dict()

    base_dir = os.path.join(Config.keywords_dir(), 'yearly_clusters')
    create_dir_if_not_exists(base_dir)

    keys_by_year = sorted(keywords.keys(), key=lambda x: x.start_date)
    for year, files in groupby(keys_by_year, key=lambda x: x.start_date.year):
        print(f'Creating clusters for {year}')
        all_keywords = sorted(
            set(chain(*[keywords[file].keywords for file in files]))
        )
        # import pdb; pdb.set_trace()
        cluster_lookup, keyword_clusters = cluster(all_keywords)
        with open(os.path.join(base_dir, f'{year}.pickle'), 'wb') as dump_file:
            pickle.dump((cluster_lookup, keyword_clusters), dump_file,
                        protocol=pickle.HIGHEST_PROTOCOL)

        # keyword_clusters_by_year[year] = (cluster_lookup, keyword_clusters)


def write_yearly_cluster_matches():
    def get_file_year(file_name):
        return int(get_file_name_without_ext(file_name))

    base_dir = os.path.join(Config.keywords_dir(), 'yearly_matches')
    create_dir_if_not_exists(base_dir)

    cluster_files = glob(
        os.path.join(Config.keywords_dir(), 'yearly_clusters', '*.pickle')
    )
    for prev_file_n, curr_file_n in window(sorted(cluster_files,
                                                  key=get_file_year)):
        print(f'Starting for {get_file_year(prev_file_n)}-{get_file_year(curr_file_n)}')
        with open(prev_file_n, 'rb') as prev_file:
            word_prev_cluster, prev_clusters = pickle.load(prev_file)
        with open(curr_file_n, 'rb') as curr_file:
            word_cluster, curr_clusters = pickle.load(curr_file)

        matches = find_cluster_matches_semantic(
            word_prev_cluster, prev_clusters,
            word_cluster, curr_clusters
        )
        matches_file_name = os.path.join(
            base_dir, f'matches_{get_file_year(curr_file_n)}.pickle'
        )
        with open(matches_file_name, 'wb') as matches_file:
            pickle.dump(matches, matches_file)


if __name__ == '__main__':
    write_yearly_cluster_matches()
