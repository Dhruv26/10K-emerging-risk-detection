import os

from nltk.tokenize import word_tokenize
from top2vec import Top2Vec
from tqdm import tqdm

from config import Config
from industry_groupings import get_industry_groups, group_by_sic_division
from report_parser import report_info_from_risk_path
from utils import get_risk_filenames


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
    model = Top2Vec(docs, document_ids=doc_ids, speed='deep-learn', workers=16)

    model_path = os.path.join(Config.top2vec_models_dir(),
                              'top2vec_model_deep_with_doc_ids')
    model.save(model_path)
    print(f'Saved model to {model_path}')


def _run_industry_wise():
    industry_groups = group_by_sic_division()
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
