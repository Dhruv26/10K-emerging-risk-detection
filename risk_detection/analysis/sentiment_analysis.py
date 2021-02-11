import os
from typing import Dict, Sequence
import logging

import numpy as np
import pandas as pd
from finbert.finbert import predict as _predict
from pytorch_pretrained_bert import BertForSequenceClassification
from tqdm import tqdm

from config import Config
from risk_detection.preprocessing.report_parser import (
    report_info_from_risk_path
)
from risk_detection.utils import (
    get_word_sentiment_df, get_risk_filenames, create_dir_if_not_exists,
    get_file_name_without_ext
)

logging.getLogger().setLevel(logging.ERROR)


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


if __name__ == '__main__':
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
