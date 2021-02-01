import os
from glob import glob
from pathlib import Path

import pandas as pd
from nltk.tokenize import word_tokenize
from pkg_resources import resource_stream
from top2vec import Top2Vec
from tqdm import tqdm

from config import Config


class IndustryGroup:
    def __init__(self, sic, ciks):
        self.sic = sic
        self.ciks = ciks
        self.filenames = IndustryGroup._get_filenames_for_ciks(self.ciks)

    def get_corpus(self):
        corpus = dict()
        for risk_filename in self.filenames:
            with open(risk_filename) as risk_file:
                docu = risk_file.read()
            if len(word_tokenize(docu)) > 100:
                corpus[risk_filename] = docu
        return corpus

    def create_topic(self):
        corpus = self.get_corpus()
        doc_ids, docs = list(zip(*corpus.items()))
        return Top2Vec(docs, document_ids=doc_ids,
                       speed='deep-learn', workers=16)

    def __repr__(self):
        return f'SIC: {self.sic}, CIKs: {len(self.ciks)}'

    @staticmethod
    def _get_filenames_for_ciks(ciks):
        risk_dir = Config.risk_dir()
        filenames = list()
        for cik in ciks:
            files = glob(os.path.join(os.path.join(risk_dir, str(cik)),
                                      '*.txt'))
            if files:
                filenames.extend(files)
        return filenames


def _get_industry_groups(sic_col='sic2'):
    df = pd.read_csv(resource_stream(
        'preprocessing', os.path.join('resources', 'cik_industry.csv')
    ))
    group = df[['cik', sic_col]].groupby(sic_col)
    res = list()
    for sic, group_df in group:
        ciks = set(group_df['cik'].unique())
        res.append(IndustryGroup(sic, ciks))
    return res


def get_corpus(path=Config.risk_dir()):
    """Creates a corpus from all files in the passed path"""
    risk_dir = Path(path)
    risk_files = list(risk_dir.rglob('*.txt'))

    corpus = dict()
    for risk_file in tqdm(risk_files):
        docu = risk_file.read_text()
        if len(word_tokenize(docu)) > 100:
            corpus[risk_file] = docu

    return corpus


def _run_all():
    risk_dir = os.path.join(Config.risk_dir())
    print(f'Reading files from {risk_dir}')
    corpus = get_corpus(risk_dir)
    print(f'Read {len(corpus)} files.')

    docs, doc_ids = list(zip(*corpus.items()))
    model = Top2Vec(docs, document_ids=doc_ids, speed='deep-learn', workers=16)

    model_path = os.path.join(Config.top2vec_models_dir(),
                              'top2vec_model_deep_with_doc_ids')
    model.save(model_path)
    print(f'Saved model to {model_path}')


def _run_industry_wise():
    industry_groups = _get_industry_groups()
    for industry_group in tqdm(industry_groups):
        if len(industry_group.ciks) < 20:
            continue

        try:
            model = industry_group.create_topic()
            model_path = os.path.join(Config.top2vec_models_dir(),
                                      'industry_wise',
                                      f'{industry_group.sic}_model')
            model.save(model_path)
        except:
            pass


if __name__ == '__main__':
    # _run_all()
    _run_industry_wise()
