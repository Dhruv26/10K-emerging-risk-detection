import logging
import os
import re
import string
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
from keybert import KeyBERT
from nltk.corpus import stopwords
from pke.unsupervised import TextRank
from rake_nltk.rake import Rake
from tqdm import tqdm

from config import Config
from risk_detection.preprocessing.report_parser import (
    report_info_from_risk_path, ReportInfo
)
from risk_detection.utils import (
    get_risk_filenames, create_dir_if_not_exists, get_file_name_without_ext
)

logging.getLogger().setLevel(logging.ERROR)


class KeywordExtractor:
    _key_bert: KeyBERT
    _rake: Rake
    _text_rank: TextRank

    def __init__(self, min_length=1, max_length=3, num_keywords=30):
        self.min_length = min_length
        self.max_length = max_length
        self.num_keywords = num_keywords

        self._key_bert = KeyBERT()
        self._rake = Rake(stopwords=stopwords.words('english'),
                          punctuations=string.punctuation,
                          min_length=min_length, max_length=max_length)
        self._text_rank = TextRank()

    def extract_using_rake(self, text: str) -> List[str]:
        self._rake.extract_keywords_from_text(text)
        return self._rake.get_ranked_phrases()[:self.num_keywords]

    def extract_using_bert(self, text: str) -> List[str]:
        return self._key_bert.extract_keywords(
            docs=text,
            keyphrase_ngram_range=(self.min_length, self.max_length),
            top_n=self.num_keywords, diversity=0.3
        )

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
    sentiment_df: pd.DataFrame

    def __init__(self, keywords: List[str], report_info: ReportInfo):
        self.keywords = keywords
        self.report_info = report_info
        self.sentiment_df = self._load_sentiment_file()

    def _load_sentiment_file(self):
        file_name = get_file_name_without_ext(self.report_info.get_file_name())
        sentiment_file = os.path.join(
            Config.risk_sentiment_dir(), str(self.report_info.cik),
            f'{file_name}.pickle'
        )
        return pd.read_pickle(sentiment_file)

    def get_negative_keywords(self) -> List[str]:
        neg_keywords = list()
        for keyword in self.keywords:
            occurrences = self.sentiment_df[
                self.sentiment_df['sentence'].str.contains(re.escape(keyword),
                                                           case=False)
            ]
            if (occurrences.sentiment_score < 0).any():
                if not (occurrences.prediction == 'negative').any():
                    import pdb; pdb.set_trace()
                neg_keywords.append(keyword)

        return neg_keywords


def write_keywords():
    keywords_dir = Config.text_rank_keywords_dir()
    create_dir_if_not_exists(keywords_dir)
    print(f'Writing found keywords to {keywords_dir}')

    keyword_extractor = KeywordExtractor(min_length=1, max_length=3,
                                         num_keywords=100)

    for risk_file_path in tqdm(get_risk_filenames()):
        report_info = report_info_from_risk_path(risk_file_path)
        cik_dir = os.path.join(keywords_dir, str(report_info.cik))
        create_dir_if_not_exists(cik_dir)

        base_filename = os.path.join(
            cik_dir, get_file_name_without_ext(report_info.get_file_name())
        )

        try:
            text = risk_file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            continue

        tr_keywords = keyword_extractor.extract_using_text_rank(text)
        with open(base_filename, 'w+', encoding='utf-8') as keywords_file:
            keywords_file.write('\n'.join(tr_keywords))


def get_neg_keywords():
    def _date_parser(date_str: str) -> datetime:
        return datetime.strptime(date_str, '%Y-%m-%d')

    def from_keyword_path(pth):
        cik = int(pth.parent.name)
        file_name = pth.name
        s_dt, e_dt, f_name, rest = file_name.split('_')
        return ReportInfo(cik, _date_parser(s_dt),
                          _date_parser(e_dt),
                          '10-K', f'{f_name}.txt')

    path = Path(r'C:\machine_learning\10K-emerging-risk-detection\models\keywords\3453')
    for doc_id in path.glob('*.txt'):
        with open(os.path.join(path, doc_id), 'r', encoding='utf-8') as f:
            keywords = [k.strip() for k in f]

        report_info = from_keyword_path(doc_id)
        keywords_obj = Keywords(keywords, report_info)
        neg_keys = keywords_obj.get_negative_keywords()
        import pdb; pdb.set_trace()


if __name__ == '__main__':
    write_keywords()
