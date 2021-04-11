import os
import pickle
from collections import defaultdict, Counter
from glob import glob
from itertools import chain, groupby
from typing import Dict, List, Tuple

import numpy as np
import sentence_transformers.util
from fuzzywuzzy import process
from sentence_transformers import SentenceTransformer
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
                    ) // 2
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


def find_new_clusters(cik, print_matched=False, print_unmatched=True):
    import risk_detection.analysis.keyword_extraction
    keywords = risk_detection.analysis.keyword_extraction.get_keywords(cik)
    sorted_keywords = sorted(keywords,
                             key=lambda keys: keys.report_info.start_date)
    for prev_keywords, curr_keywords in window(sorted_keywords):
        print(f'-------{curr_keywords.report_info.start_date.year}-------')
        word_prev_cluster, prev_clusters = prev_keywords.cluster()
        word_cluster, curr_clusters = curr_keywords.cluster()

        matches = find_cluster_matches_semantic(
            word_prev_cluster, prev_clusters, word_cluster, curr_clusters
        )
        if print_matched:
            for curr_cl, prev_cl in matches.items():
                print(f'{curr_clusters[curr_cl]} --> {prev_clusters[prev_cl]}')
        unmatched_clusters = curr_clusters.keys() - matches.keys()
        if print_unmatched:
            for curr_cl in unmatched_clusters:
                print(f'{curr_clusters[curr_cl]}')


def read_yearly_cluster(year):
    filename = os.path.join(Config.keywords_dir(), 'yearly_clusters',
                            f'{year}.pickle')
    with open(filename, 'rb') as f:
        return pickle.load(f)


def analyze_yearly_clusters(print_matched=False, print_unmatched=True):
    keywords_dir = os.path.join(Config.keywords_dir(), 'yearly_clusters')
    matches_template = os.path.join(Config.keywords_dir(), 'yearly_matches',
                                    'matches_{}.pickle')

    yearly_filenames = glob(os.path.join(keywords_dir, '*.pickle'))

    num_clusters = list()
    num_words_in_clusters = list()
    num_matches = list()

    sorted_filenames = sorted(yearly_filenames,
                              key=lambda k: int(get_file_name_without_ext(k)))
    for prev, curr in window(sorted_filenames):
        if print_matched or print_unmatched:
            print(f'-------{get_file_name_without_ext(curr)}-------')
        with open(prev, 'rb') as f:
            prev_word_cl, prev_clusters = pickle.load(f)
        with open(curr, 'rb') as f:
            curr_word_cl, curr_clusters = pickle.load(f)

        num_clusters.append(len(curr_clusters))
        num_words_in_clusters.extend([len(v) for v in curr_clusters.values()])
        matches_filename = matches_template.format(
            get_file_name_without_ext(curr)
        )
        with open(matches_filename, 'rb') as matches_file:
            matches = pickle.load(matches_file)
        num_matches.append(len(matches))

        if print_matched:
            for curr_cl, prev_cl in matches.items():
                print(f'{curr_clusters[curr_cl]} --> {prev_clusters[prev_cl]}')
        unmatched_clusters = curr_clusters.keys() - matches.keys()
        if print_unmatched:
            for curr_cl in unmatched_clusters:
                print(f'{curr_clusters[curr_cl]}')

    print(f'Average Number of Yearly Clusters: '
          f'{sum(num_clusters) / len(num_clusters)}')
    print(f'Average Number of Words in Yearly Clusters: '
          f'{sum(num_words_in_clusters) / len(num_words_in_clusters)}')
    print(f'Average number of Yearly Matches Found: '
          f'{sum(num_matches) / len(num_matches)}')


if __name__ == '__main__':
    """
    1750 -> Aar Corp is primarily in the business of aircraft & parts.
    """
    # find_new_clusters(1750)
    """
    3116 -> Akorn Inc is primarily in the business of pharmaceutical 
    preparations. Akorn manufactures ophthalmology and dermatology products 
    as well as injectables.
    """
    # find_new_clusters(3116) ### Healthcare
    """ 
    2488 -> Advanced Micro Devices Inc is primarily in the business of 
    semiconductors & related devices.
    """
    # find_new_clusters(2488)
    """
    28917 -> Dillard's Inc is primarily in the business of retail-department 
    stores. Dillard's Inc is an American fashion apparel, cosmetics and home 
    furnishings retailer.
    """
    # find_new_clusters(28917)
    analyze_yearly_clusters(False, False)
