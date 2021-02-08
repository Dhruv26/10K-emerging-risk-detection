from pathlib import Path
from typing import List, Sequence

from nltk.tokenize import word_tokenize
from top2vec import Top2Vec

from report_parser import report_info_from_risk_path
from utils import (get_risk_filenames_for_ciks, get_company_industry_mapping,
                   get_sik_industry_name_mapping)


class IndustryGroup:
    sic: str
    ciks: Sequence[int]
    filenames: Sequence[Path]

    def __init__(self, sic: str, ciks: Sequence[int]):
        self.sic = sic
        self.ciks = ciks
        self.filenames = get_risk_filenames_for_ciks(self.ciks)

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

    def __repr__(self):
        return f'Group: {self.sic}, CIKs: {len(self.ciks)}'


def get_industry_groups(sic_col: str = 'sic2') -> List[IndustryGroup]:
    df = get_company_industry_mapping()
    group = df[['cik', sic_col]].groupby(sic_col)
    res = list()
    for sic, group_df in group:
        ciks = set(group_df['cik'].unique())
        res.append(IndustryGroup(sic, ciks))
    return res


def group_by_sic_division() -> List[IndustryGroup]:
    cols = ['SIC DIVISION', 'MOST SPECIFIC SIC CODE']
    sic_grouping_df = get_sik_industry_name_mapping()[cols]
    cik_sic_df = get_company_industry_mapping()[['cik', 'sic']]

    res = list()
    for grp_name, grp_df in sic_grouping_df.groupby('SIC DIVISION'):
        sics = set(map(
            int, filter(lambda x: x.isdigit(),
                        grp_df['MOST SPECIFIC SIC CODE'].unique())
        ))
        # ciks should be unique. Calling Series#unique as a safety net
        ciks = cik_sic_df[cik_sic_df['sic'].isin(sics)]['cik'].unique()
        if ciks.size > 0:
            res.append(IndustryGroup(grp_name, ciks))
    return res


if __name__ == '__main__':
    group_by_sic_division()
