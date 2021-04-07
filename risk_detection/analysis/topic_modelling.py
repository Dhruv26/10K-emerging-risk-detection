import os
from collections import defaultdict
from typing import Sequence, List

from nltk.tokenize import word_tokenize
from top2vec import Top2Vec
from tqdm import tqdm

from config import Config
from industry.groupings import IndustryGroupCreator
from risk_detection.analysis.doc_processor import RiskSectionCleaner
from risk_detection.analysis.sentiment_analysis import (
    create_sentiment_analyser, ContextualizedWordSentiment
)
from risk_detection.preprocessing.report_parser import (
    report_info_from_risk_path, ReportInfo
)
from risk_detection.utils import (
    get_risk_filenames, create_dir_if_not_exists
)


class Topic:
    topic_words: Sequence[str]
    word_scores: Sequence[str]
    topic_num: int
    doc_ids: Sequence[str]
    sentiment_analyser: ContextualizedWordSentiment

    def __init__(self, topic_words: Sequence[str], word_scores: Sequence[str],
                 topic_num: int, doc_ids: Sequence[str]):
        self.topic_words = topic_words
        self.word_scores = word_scores
        self.topic_num = topic_num
        self.doc_ids = doc_ids

        report_infos = [ReportInfo.from_doc_id(doc_id) for doc_id in doc_ids]
        self.sentiment_analyser = create_sentiment_analyser(report_infos)

    def get_negative_terms(self) -> List[str]:
        """
        Finds the negative terms in a topic by going through the
        documents in this topic and checking which terms occur in a
        negative context.
        A term is in a negative context if the majority of sentences
        in which it occurs has a negative sentiment.

        :return: List of negative terms in the topic
        """
        sentiment = (self.sentiment_analyser
                     .get_sentiment_for_words(self.topic_words))
        return [word
                for word, sentiment_score in sentiment.items()
                if sentiment_score < 0]


def get_corpus():
    """Creates a corpus from all files in the passed path"""
    risk_files = get_risk_filenames()

    corpus = dict()
    for risk_file in tqdm(risk_files):
        docu = risk_file.read_text(encoding='utf-8')
        if len(word_tokenize(docu)) > 100:
            report_info = report_info_from_risk_path(risk_file)
            corpus[report_info.get_document_id()] = docu

    return corpus


def run_all():
    """
    Creates a topic model for the entire corpus and saves it to disk.
    """
    print(f'Reading files from {Config.risk_dir()}')
    corpus = get_corpus()
    print(f'Read {len(corpus)} files.')

    doc_ids, docs = list(zip(*corpus.items()))
    model = Top2Vec(docs, document_ids=doc_ids, tokenizer=RiskSectionCleaner(),
                    keep_documents=False, speed='learn', workers=24)

    model_path = os.path.join(Config.top2vec_models_dir(),
                              'top2vec_model_phrases')
    model.save(model_path)
    print(f'Saved model to {model_path}')


def run_industry_wise():
    industry_groups = IndustryGroupCreator.create_by_sic_division()
    for industry_group in tqdm(industry_groups):
        if len(industry_group.ciks) < 20:
            continue
        model = industry_group.create_topic()
        model_path = os.path.join(Config.top2vec_models_dir(),
                                  'sic_group_industry_wise',
                                  f'{industry_group.sic}_model')
        model.save(model_path)


def run_yearly():
    """
    Creates yearly topic models and saves it to disk.
    """
    print(f'Reading files from {Config.risk_dir()}')
    corpus = get_corpus()
    yearly_doc_ids = defaultdict(list)
    for k in corpus.keys():
        yearly_doc_ids[ReportInfo.from_doc_id(k).start_date.year].append(k)
    print(f'Read {len(corpus)} files.')

    base_dir = os.path.join(Config.top2vec_models_dir(), 'yearly_models')
    create_dir_if_not_exists(base_dir)
    print(f'Storing yearly models to {base_dir}.')

    for year, doc_ids in tqdm(yearly_doc_ids.items(),
                              total=len(yearly_doc_ids)):
        yearly_corpus = [corpus[d] for d in doc_ids]
        model = Top2Vec(documents=yearly_corpus, document_ids=doc_ids,
                        tokenizer=RiskSectionCleaner(), keep_documents=False,
                        speed='learn', workers=24)
        model.save(os.path.join(base_dir, f'{year}_topics'))


def _test_topics():
    def _get_noun_phrases(text):
        pass
    model_path = os.path.join(Config.top2vec_models_dir(),
                              'top2vec_model_with_doc_ids')
    model = Top2Vec.load(model_path)
    print('Creating topics.')
    for topic_size, topic_num in zip(*model.get_topic_sizes()):
        if topic_num < 200:
            continue
        _, doc_ids = model.search_documents_by_topic(
            topic_num, num_docs=topic_size, return_documents=False
        )
        topic_words = model.topic_words[topic_num]
        word_scores = model.topic_word_scores[topic_num]
        topic = Topic(topic_words, word_scores, topic_num, doc_ids)
        neg_words = topic.get_negative_terms()
        a = 1


if __name__ == '__main__':
    # run_all()
    # run_industry_wise()
    run_yearly()
