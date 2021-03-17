from collections import defaultdict
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

embedder = SentenceTransformer('stsb-roberta-base')


def cluster(corpus: List[str]) -> Dict[int, List[str]]:
    corpus_embeddings = embedder.encode(corpus)
    # Normalize embeddings to unit length
    corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings,
                                                           axis=1,
                                                           keepdims=True)

    clustering_model = AgglomerativeClustering(n_clusters=None,
                                               distance_threshold=1.5)
    clustering_model.fit(corpus_embeddings)

    clustered_sentences = defaultdict(list)
    for sentence_id, cluster_id in enumerate(clustering_model.labels_):
        clustered_sentences[cluster_id].append(corpus[sentence_id])

    return clustered_sentences
