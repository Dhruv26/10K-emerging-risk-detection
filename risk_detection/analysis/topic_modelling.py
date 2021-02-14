import os
from typing import Sequence, List

from nltk.tokenize import word_tokenize
from top2vec import Top2Vec
from tqdm import tqdm

from config import Config
from industry.groupings import IndustryGroupCreator
from risk_detection.preprocessing.report_parser import (
    report_info_from_risk_path, ReportInfo
)
from risk_detection.analysis.sentiment_analysis import (
    create_sentiment_analyser, ContextualizedWordSentiment
)
from risk_detection.analysis.doc_processor import RiskSectionCleaner
from risk_detection.utils import (
    get_risk_filenames, get_risk_section_sentiment_files
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
        # TODO: Implement
        pass


def get_corpus():
    """Creates a corpus from all files in the passed path"""
    risk_files = get_risk_filenames()

    corpus = dict()
    for risk_file in tqdm(risk_files):
        docu = risk_file.read_text()
        if len(word_tokenize(docu)) > 100:
            report_info = report_info_from_risk_path(risk_file)
            corpus[report_info.get_document_id()] = docu

    return corpus


def _run_all():
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


def _run_industry_wise():
    industry_groups = IndustryGroupCreator.create_by_sic_division()
    for industry_group in tqdm(industry_groups):
        if len(industry_group.ciks) < 20:
            continue
        model = industry_group.create_topic()
        model_path = os.path.join(Config.top2vec_models_dir(),
                                  'sic_group_industry_wise',
                                  f'{industry_group.sic}_model')
        model.save(model_path)


if __name__ == '__main__':
    # _run_all()
    _run_industry_wise()
    """
    def _get_noun_phrases(text):    pass
    model_path = os.path.join(Config.top2vec_models_dir(),
                              'top2vec_model_with_doc_ids')
    model = Top2Vec.load(model_path)
    doc_ids = model.search_documents_by_topic(300,
                                              num_docs=model.topic_sizes[300],
                                              return_documents=False)
    for topic_size, topic_num in model.get_topic_sizes():
        import pdb; pdb.set_trace()
    
    # TODO: Change to above
    risk_sentiment_files = get_risk_section_sentiment_files()
    topic = Topic([], [], 1, [
        report_info_from_risk_path(x).get_document_id()
        for x in risk_sentiment_files[:10]
    ])
    topic.get_negative_terms()
    """
