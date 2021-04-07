import itertools
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Sequence, Dict, Tuple

from nltk.tokenize import word_tokenize
from top2vec import Top2Vec

from config import Config
from risk_detection.analysis.clustering import cluster
from risk_detection.analysis.keyword_extraction import Keywords, get_keywords
from risk_detection.preprocessing.report_parser import (
    report_info_from_risk_path
)
from risk_detection.utils import (
    get_risk_filenames_for_ciks, get_company_industry_mapping,
    get_sik_industry_name_mapping, create_dir_if_not_exists
)


class IndustryGroup:
    sic_category: Optional[str]
    sic_grp_name: Optional[str]
    cik_codes: Sequence[int]
    sic_codes: Sequence[int]
    sic_code_names: Sequence[str]

    filenames: Sequence[Path]

    def __init__(self, sic_category: Optional[str],
                 sic_grp_name: Optional[str], cik_codes: Sequence[int],
                 sic_codes: Sequence[int], sic_code_names: Sequence[str]):
        self.sic_category = sic_category
        self.sic_grp_name = sic_grp_name
        self.cik_codes = cik_codes
        self.sic_codes = sic_codes
        self.sic_code_names = sic_code_names

        self.filenames = get_risk_filenames_for_ciks(self.cik_codes)

    def get_corpus(self):
        corpus = dict()
        for risk_filename in self.filenames:
            docu = risk_filename.read_text()
            if len(word_tokenize(docu)) > 100:
                report_info = report_info_from_risk_path(risk_filename)
                corpus[report_info.get_document_id()] = docu
        return corpus

    def create_topic(self):
        corpus = self.get_corpus()
        doc_ids, docs = list(zip(*corpus.items()))
        return Top2Vec(docs, document_ids=doc_ids,
                       speed='deep-learn', workers=16)

    def _organize_keywords_by_year(self) -> Dict[int, Keywords]:
        keywords = defaultdict(list)
        for cik in self.cik_codes:
            for key in get_keywords(cik):
                keywords[key.report_info.start_date.year].append(key)
        return keywords

    def cluster_keywords(self) -> Dict[int, Tuple]:
        yearly_clusters = dict()

        keywords = self._organize_keywords_by_year()
        for year, keys in keywords.items():
            all_keys = list(set(itertools.chain.from_iterable([
                k.keywords for k in keys
            ])))
            yearly_clusters[year] = cluster(all_keys)

        return yearly_clusters

    def __repr__(self):
        return f'Group: {self.sic_grp_name}, CIKs: {len(self.cik_codes)}'


class IndustryGroupCreator:
    cik_sic_df = get_company_industry_mapping()
    sic_name_df = get_sik_industry_name_mapping()

    @staticmethod
    def create_by_sic_division() -> List[IndustryGroup]:
        industry_groups = list()

        groups = IndustryGroupCreator.sic_name_df.groupby('SIC DIVISION')
        for sic_grp_name, grp_df in groups:
            grp_name = grp_df.iloc[0]['CATEGORY DESCRIPTION']

            names = grp_df.iloc[1:]['CATEGORY DESCRIPTION'].values
            sic_code = set(map(int, grp_df.iloc[1:]['MOST SPECIFIC SIC CODE']))

            # ciks should be unique. Calling Series#unique as a safety net
            ciks = IndustryGroupCreator.cik_sic_df[
                IndustryGroupCreator.cik_sic_df['sic'].isin(sic_code)
            ]['cik'].unique()

            industry_groups.append(
                IndustryGroup(sic_grp_name, grp_name, ciks, sic_code, names)
            )

        return industry_groups

    @staticmethod
    def create_by_sic_char_length(sic_c: str = 'sic2') -> List[IndustryGroup]:
        group = IndustryGroupCreator.cik_sic_df[['cik', sic_c]].groupby(sic_c)
        res = list()
        for sic, group_df in group:
            ciks = set(group_df['cik'].unique())
            # TODO: Fix
            # res.append(IndustryGroup(sic, ciks))
        return res


if __name__ == '__main__':
    base_dir = os.path.join(Config.keywords_dir(), 'industry_groups',
                            'sic_groups')
    create_dir_if_not_exists(base_dir)

    groups = IndustryGroupCreator.create_by_sic_division()
    for group in groups:
        industry_yearly_clusters = group.cluster_keywords()
        if industry_yearly_clusters:
            with open(os.path.join(base_dir, f'{group.sic_category}.pickle'),
                      'wb') as f:
                pickle.dump(industry_yearly_clusters, f,
                            protocol=pickle.HIGHEST_PROTOCOL)
