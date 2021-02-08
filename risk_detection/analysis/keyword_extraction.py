import os
import string
from typing import List

from keybert import KeyBERT
from nltk.corpus import stopwords
from rake_nltk.rake import Rake
from tqdm import tqdm

from config import Config
from risk_detection.preprocessing.report_parser import (
    report_info_from_risk_path
)
from risk_detection.utils import get_risk_filenames, create_dir_if_not_exists


class KeywordExtractor:
    def __init__(self, min_length=1, max_length=3, num_keywords=30):
        self.min_length = min_length
        self.max_length = max_length
        self.num_keywords = num_keywords

        self._key_bert = KeyBERT()
        self._rake = Rake(stopwords=stopwords.words('english'),
                          punctuations=string.punctuation,
                          min_length=min_length, max_length=max_length)

    def extract_using_rake(self, text: str) -> List[str]:
        self._rake.extract_keywords_from_text(text)
        return self._rake.get_ranked_phrases()[:self.num_keywords]

    def extract_using_bert(self, text: str) -> List[str]:
        return self._key_bert.extract_keywords(
            docs=text,
            keyphrase_ngram_range=(self.min_length, self.max_length),
            top_n=self.num_keywords, diversity=0.3
        )


if __name__ == '__main__':
    keywords_dir = os.path.join(Config.models_dir(), 'keywords')
    create_dir_if_not_exists(keywords_dir)
    print(f'Writing found keywords to {keywords_dir}')

    keyword_extractor = KeywordExtractor(min_length=1, max_length=3)

    for risk_file_path in tqdm(get_risk_filenames()):
        report_info = report_info_from_risk_path(risk_file_path)
        cik_dir = os.path.join(keywords_dir, str(report_info.cik))
        create_dir_if_not_exists(cik_dir)

        text = risk_file_path.read_text()
        bert_keywords = keyword_extractor.extract_using_bert(text)
        rake_keywords = keyword_extractor.extract_using_rake(text)

        base_filename = report_info.get_file_name()
        with open(base_filename + '_bert', 'w+') as bert_keywords_file:
            bert_keywords_file.writelines(bert_keywords)
        with open(base_filename + '_rake', 'w+') as rake_keywords_file:
            rake_keywords_file.writelines(rake_keywords)
