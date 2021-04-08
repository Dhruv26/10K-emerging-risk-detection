import glob
import logging
import os
import re
import string
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pke.unsupervised import TextRank
from rake_nltk.rake import Rake
from tqdm import tqdm

from config import Config
from risk_detection.analysis.clustering import cluster
from risk_detection.preprocessing.report_parser import (
    report_info_from_risk_path, ReportInfo
)
from risk_detection.utils import (
    get_risk_filenames, create_dir_if_not_exists, get_file_name_without_ext,
    window
)

logging.getLogger().setLevel(logging.ERROR)


class KeywordExtractor:
    _rake: Rake
    _text_rank: TextRank

    def __init__(self, min_length=1, max_length=3, num_keywords=30):
        self.min_length = min_length
        self.max_length = max_length
        self.num_keywords = num_keywords

        self._rake = Rake(stopwords=stopwords.words('english'),
                          punctuations=string.punctuation,
                          min_length=min_length, max_length=max_length)
        self._text_rank = TextRank()

    def extract_using_rake(self, text: str) -> List[str]:
        self._rake.extract_keywords_from_text(text)
        return self._rake.get_ranked_phrases()[:self.num_keywords]

    def extract_using_text_rank(self, text: str,
                                window: int = 2,
                                top_percent: int = 0.5) -> List[str]:
        pos = {'NOUN', 'PROPN', 'ADJ'}
        self._text_rank.load_document(input=text,
                                      language='en',
                                      normalization=None)
        self._text_rank.candidate_weighting(window=window, pos=pos,
                                            top_percent=top_percent)
        return [key
                for key, _ in self._text_rank.get_n_best(n=self.num_keywords)]


class Keywords:
    keywords: List[str]
    report_info: ReportInfo
    neg_keywords: Dict[str, pd.DataFrame]

    def __init__(self, keywords: List[str], report_info: ReportInfo):
        self.keywords = keywords
        self.report_info = report_info
        sentiment_df = self._load_sentiment_file(self.report_info)
        self.neg_keywords = self._get_negative_keywords(keywords, sentiment_df)

    def get_keywords(self) -> List[str]:
        return self.keywords

    def get_negative_keywords(self) -> List[str]:
        return list(self.neg_keywords.keys())

    def cluster(self) -> Tuple[Dict[str, int], Dict[int, List[str]]]:
        return cluster(self.keywords)

    def cluster_neg(self) -> Tuple[Dict[str, int], Dict[int, List[str]]]:
        return cluster(list(self.neg_keywords.keys()))

    @staticmethod
    def _load_sentiment_file(report_info: ReportInfo) -> pd.DataFrame:
        file_name = get_file_name_without_ext(report_info.get_file_name())
        sentiment_file = os.path.join(
            Config.risk_sentiment_dir(), str(report_info.cik),
            f'{file_name}.pickle'
        )
        return pd.read_pickle(sentiment_file)

    @staticmethod
    def _get_negative_keywords(keywords: List[str],
                               sentiment_df: pd.DataFrame) \
            -> Dict[str, pd.DataFrame]:
        neg_keywords = dict()

        for keyword in keywords:
            occurrences = sentiment_df[
                sentiment_df['sentence'].str.contains(re.escape(keyword),
                                                      case=False)
            ]
            if (occurrences.prediction != 'positive').any():
                neg_keywords[keyword] = occurrences

        return neg_keywords


def _write_keywords_for_risk_file(risk_file_path):
    keywords_dir = Config.text_rank_keywords_dir()
    report_info = report_info_from_risk_path(risk_file_path)
    cik_dir = os.path.join(keywords_dir, str(report_info.cik))
    create_dir_if_not_exists(cik_dir)

    base_filename = os.path.join(
        cik_dir, get_file_name_without_ext(report_info.get_file_name())
    )

    try:
        text = risk_file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        return

    keyword_extractor = KeywordExtractor(min_length=1, max_length=3,
                                         num_keywords=100)
    tr_keywords = keyword_extractor.extract_using_text_rank(text)
    with open(base_filename + '.txt', 'w+', encoding='utf-8') as keywords_file:
        keywords_file.write('\n'.join(tr_keywords))


def write_keywords():
    keywords_dir = Config.text_rank_keywords_dir()
    create_dir_if_not_exists(keywords_dir)
    print(f'Writing found keywords to {keywords_dir}')

    risk_files = get_risk_filenames()
    with ProcessPoolExecutor(max_workers=4) as executor:
        tasks = [executor.submit(_write_keywords_for_risk_file, risk_file)
                 for risk_file in risk_files]

        for task in tqdm(as_completed(tasks), total=len(tasks)):
            pass


def get_keywords_for(keyword_path: Path) -> Keywords:
    return keywords_from_file(keyword_path)


@lru_cache(maxsize=128)
def generate_keywords(keyword_path: Path) -> Keywords:
    report_info = report_info_from_risk_path(keyword_path)
    risk_file_name = os.path.join(Config.risk_dir(), str(report_info.cik),
                                  report_info.get_file_name())
    with open(risk_file_name, 'r', encoding='utf-8') as risk_file:
        text = risk_file.read()

    extractor = KeywordExtractor(num_keywords=100)
    return Keywords(extractor.extract_using_text_rank(text), report_info)


def clean_keyword(keyword):
    tokens_to_remove = {'other', 'others', 'such', 'certain'}

    without_puncts = keyword.translate(
        str.maketrans('', '', string.punctuation)
    ).strip()
    tokens = word_tokenize(without_puncts)
    tags = pos_tag(tokens)
    first_word, first_tag = tags[0]
    last_word, last_tag = tags[-1]

    if first_tag == 'JJ' and first_word in tokens_to_remove:
        tags = tags[1:]
    if last_tag == 'JJ' and last_word in tokens_to_remove:
        tags = tags[:-1]

    return ' '.join(tk for tk, _ in tags) if tags else ''


def simple_clean_keyword(keyword):
    tokens_to_remove = {'other', 'others', 'such', 'certain'}

    without_puncts = keyword.translate(
        str.maketrans('', '', string.punctuation)).strip()
    if not without_puncts:
        return ''

    tokens = word_tokenize(without_puncts)
    first_word = tokens[0]
    last_word = tokens[-1]

    if first_word in tokens_to_remove:
        tokens = tokens[1:]
    if last_word in tokens_to_remove:
        tokens = tokens[:-1]

    return ' '.join(tokens) if tokens else ''


def keywords_from_file(keyword_file: Union[str, Path]) -> Keywords:
    if isinstance(keyword_file, str):
        keyword_file = Path(keyword_file)

    with open(keyword_file, 'r', encoding='utf-8') as key_f:
        keys = key_f.read()

    if not keys:
        raise ValueError(f'No keywords found for {keyword_file}')

    cleaned = set()
    # filter(lambda k: k, map(simple_clean_keyword, keys.split('\n')))
    for k in keys.split('\n'):
        cl = simple_clean_keyword(k)
        if cl:
            cleaned.add(cl)

    report_info = report_info_from_risk_path(keyword_file)
    return Keywords(sorted(list(cleaned)), report_info)


def rake_keywords_from_file(keyword_file: Union[str, Path]) -> Keywords:
    if isinstance(keyword_file, str):
        keyword_file = Path(keyword_file)

    with open(keyword_file, 'r', encoding='utf-8') as key_f:
        keys = key_f.read()

    if not keys:
        raise ValueError(f'No keywords found for {keyword_file}')

    cleaned = list()
    for k in keys.split('\n'):
        cl = simple_clean_keyword(k)
        if cl:
            cleaned.append(cl)

    st, end, filename, _ = keyword_file.name.split('_', maxsplit=4)
    actual_file = Path(os.path.join(keyword_file.parent,
                                    '_'.join((st, end, filename))))
    report_info = report_info_from_risk_path(actual_file)
    return Keywords(sorted(cleaned), report_info)


def get_all_keywords() -> Dict[ReportInfo, Keywords]:
    keywords_dir = Path(Config.text_rank_keywords_dir())
    keywords = dict()
    for keyword_file in tqdm(list(keywords_dir.rglob('*.txt'))):
        try:
            keys = keywords_from_file(keyword_file)
            keywords[keys.report_info] = keys
        except ValueError:
            continue

    return keywords


def get_keywords(cik: Union[int, str]) -> List[Keywords]:
    path = os.path.join(Config.text_rank_keywords_dir(), str(cik))
    keywords = list()

    for keyword_file in glob.glob(os.path.join(path, '*.txt')):
        try:
            keywords.append(keywords_from_file(keyword_file))
        except ValueError:
            continue

    return keywords


if __name__ == '__main__':
    get_keywords(1750)
    # get_neg_keywords()
    # write_keywords()
    # path = Path(
    #     r'C:\machine_learning\10K-emerging-risk-detection\data\risk_section\12927\2007-12-31_2008-02-15_0001193125-08-032328.txt')
    # key = get_keywords(path)
    # print(key.cluster())
