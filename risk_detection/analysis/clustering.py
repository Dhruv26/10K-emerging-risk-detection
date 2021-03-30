import os
import pickle
from collections import defaultdict
from itertools import chain, groupby
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

from config import Config
from risk_detection.utils import create_dir_if_not_exists

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


if __name__ == '__main__':
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
