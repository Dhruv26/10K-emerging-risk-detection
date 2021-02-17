import logging
import os
from collections import Counter, namedtuple
from functools import partial
from typing import Callable, Dict, Sequence

import numpy as np
import pandas as pd
from finbert.finbert import predict as _predict
from gensim.utils import simple_preprocess
from pytorch_pretrained_bert import BertForSequenceClassification
from tqdm import tqdm

from config import Config
from risk_detection.preprocessing.report_parser import (
    report_info_from_risk_path, ReportInfo
)
from risk_detection.utils import (
    get_word_sentiment_df, get_risk_filenames, create_dir_if_not_exists,
    get_file_name_without_ext
)
from risk_detection.analysis.doc_processor import RiskSectionCleaner

logging.getLogger().setLevel(logging.ERROR)

sentiment = namedtuple('SentimentWordCount', ['pos', 'neut', 'neg'])
DEFAULT_TOKENIZER = RiskSectionCleaner().simple_tokenize


def _get_pos_neg_token_sets():
    pos_neg_df = get_word_sentiment_df()
    positive = pos_neg_df.loc[pos_neg_df['Positive'] != 0,
                              'Word'].str.lower().unique()
    negative = pos_neg_df.loc[
        (pos_neg_df['Negative'] != 0) | (pos_neg_df['Uncertainty'] != 0) |
        (pos_neg_df['Litigious'] != 0) | (pos_neg_df['Constraining'] != 0) |
        (pos_neg_df['Interesting'] != 0) | (pos_neg_df['Superfluous'] != 0) |
        (pos_neg_df['Harvard_IV'] != 0),
        'Word'
    ].str.lower().unique()
    return set(positive), set(negative)


class _TokenSentimentAnalysis:
    POS_TOKENS, NEG_TOKENS = _get_pos_neg_token_sets()
    TAG_POL = 'Polarity'
    TAG_SUB = 'Subjectivity'
    TAG_SENTIMENT = 'Sentiment'
    TAG_POS = 'Positive'
    TAG_NEG = 'Negative'
    EPSILON = 1e-6

    @staticmethod
    def _get_polarity(s_pos, s_neg):
        return ((s_pos - s_neg) * 1.0 /
                ((s_pos + s_neg) + _TokenSentimentAnalysis.EPSILON))

    @staticmethod
    def _get_subjectivity(s_pos, s_neg, score_li):
        return ((s_pos + s_neg) * 1.0 /
                (len(score_li) + _TokenSentimentAnalysis.EPSILON))

    @staticmethod
    def _get_score(term):
        """
        Get score for a single term.
        - +1 for positive terms.
        - -1 for negative terms.
        - 0 for others.

        :returns: int
        """
        if term in _TokenSentimentAnalysis.POS_TOKENS:
            return +1
        elif term in _TokenSentimentAnalysis.NEG_TOKENS:
            return -1
        else:
            return 0

    @staticmethod
    def get_score(terms: Sequence[str]) -> Dict[str, float]:
        """
        Get score for a list of terms.

        :type terms: An iterable of tokens
        :param terms: A list of terms to be analyzed
        :returns: dict
        """
        score_li = np.asarray([_TokenSentimentAnalysis._get_score(t)
                               for t in terms])

        s_pos = np.sum(score_li[score_li > 0])
        s_neg = -np.sum(score_li[score_li < 0])
        s_pol = _TokenSentimentAnalysis._get_polarity(s_pos, s_neg)
        s_sub = _TokenSentimentAnalysis._get_subjectivity(s_pos, s_neg,
                                                          score_li)
        return {
            _TokenSentimentAnalysis.TAG_POS: terms[score_li == 1],
            _TokenSentimentAnalysis.TAG_NEG: terms[score_li == -1],
            _TokenSentimentAnalysis.TAG_POL: s_pol,
            _TokenSentimentAnalysis.TAG_SUB: s_sub
        }


def get_token_sentiment(token: str) -> Dict[str, float]:
    return _TokenSentimentAnalysis.get_score(token)


class Weights:
    pos = 20
    neut = -1
    neg = -3


class ContextualizedWordSentiment:
    """
    Finds the sentiment for tokens based on their occurrences
    in documents.
    """
    report_infos: Sequence[ReportInfo]
    pos: Dict[str, float]
    neut: Dict[str, float]
    neg: Dict[str, float]

    def __init__(self, report_infos: Sequence[ReportInfo],
                 sentiment_word_count: sentiment):
        self.report_infos = report_infos
        self.pos = sentiment_word_count.pos
        self.neut = sentiment_word_count.neut
        self.neg = sentiment_word_count.neg

    def get_sentiment_for_words(self,
                                words: Sequence[str]) -> Dict[str, float]:
        res = dict()
        for word in words:
            num_pos = self.pos[word]
            num_neut = self.neut[word]
            num_neg = self.neg[word]
            # self._validate_word_occurrences(num_neg, num_neut, num_pos, word)
            total_occurrences = sum((num_pos, num_neut, num_neg))
            if not total_occurrences:
                res[word] = 0
                continue
            # TODO: Look into the weighting scheme
            res[word] = (((num_pos * Weights.pos) +
                          (num_neut * Weights.neut) +
                          (num_neg * Weights.neg)) / total_occurrences)

        # import pdb; pdb.set_trace()
        return res

    @staticmethod
    def _validate_word_occurrences(num_neg, num_neut, num_pos, word):
        if sum((num_pos, num_neut, num_neg)) == 0:
            raise ValueError(f'Word: {word} not in vocabulary.')


class SentimentFilesProcessor:
    def __init__(self, report_infos: Sequence[ReportInfo],
                 tokenizer: Callable[[str], Sequence[str]]):
        self.report_sentiments = (SentimentFilesProcessor
                                  ._get_sentiment_files(report_infos))
        self.tokenizer = tokenizer

    def process(self) -> sentiment:
        pos = Counter()
        neut = Counter()
        neg = Counter()

        mapping = {'positive': pos, 'neutral': neut, 'negative': neg}

        def process_row(row):
            word_count = self._get_word_count(row.sentence)
            mapping[row.prediction] += word_count
            return word_count

        for report_info, sentiment_df in self.report_sentiments.items():
            sentiment_df.apply(process_row, axis=1)

        return sentiment(pos, neut, neg)

    def _get_word_count(self, sentence) -> Dict[str, int]:
        return Counter(self.tokenizer(sentence))

    @staticmethod
    def _get_sentiment_files(report_infos: Sequence[ReportInfo]) \
            -> Dict[ReportInfo, pd.DataFrame]:
        base_dir = Config.risk_sentiment_dir()
        sentiment_dict = dict()
        for report_info in report_infos:
            file_name = get_file_name_without_ext(report_info.get_file_name())
            sentiment_file = os.path.join(
                base_dir, str(report_info.cik), f'{file_name}.pickle'
            )
            sentiment_dict[report_info] = pd.read_pickle(sentiment_file)
        return sentiment_dict


def create_sentiment_analyser(
        report_infos: Sequence[ReportInfo],
        tokenizer: Callable[[str], Sequence[str]] = DEFAULT_TOKENIZER
) -> ContextualizedWordSentiment:
    file_processor = SentimentFilesProcessor(report_infos, tokenizer)
    sentiment_word_count = file_processor.process()
    return ContextualizedWordSentiment(report_infos, sentiment_word_count)


_model_path = Config.finBERT_model_dir()
_model = BertForSequenceClassification.from_pretrained(
    _model_path, num_labels=3, cache_dir=True
)


def get_sentiment(text: str) -> pd.DataFrame:
    """
    Predicts the sentiment of each sentence in text. Relies BERT
    on the model produced by https://github.com/ProsusAI/finBERT.

    :param text: Can be a paragraph or a single sentence.
        A paragraph will be tokenized.
    :return: Pandas dataframe containing each sentence in
        text along with info about it's sentiment.
    """
    return _predict(text, _model)


def write_doc_sentiment_files():
    base_dir = Config.risk_sentiment_dir()
    create_dir_if_not_exists(base_dir)

    for risk_file in tqdm(get_risk_filenames()):
        risk_section = risk_file.read_text(encoding='utf-8')
        sentiment_df = get_sentiment(risk_section)

        report_info = report_info_from_risk_path(risk_file)
        file_path = os.path.join(base_dir, str(report_info.cik))
        create_dir_if_not_exists(file_path)

        filename = get_file_name_without_ext(report_info.get_file_name())
        sentiment_filename = os.path.join(file_path, f'{filename}.pickle')

        sentiment_df.to_pickle(path=sentiment_filename)


if __name__ == '__main__':
    write_doc_sentiment_files()
