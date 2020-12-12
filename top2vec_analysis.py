import os
from pathlib import Path

from nltk.tokenize import word_tokenize
from top2vec import Top2Vec
from tqdm import tqdm

from config import Config


def get_corpus(path=Config.risk_dir()):
    """Creates a corpus from all files in the passed path"""
    risk_dir = Path(path)
    risk_files = list(risk_dir.rglob('*.txt'))

    corpus = []
    for risk_file in tqdm(risk_files):
        docu = risk_file.read_text()
        if len(word_tokenize(docu)) > 100:
            corpus.append(docu)

    return corpus


if __name__ == '__main__':
    risk_dir = Config.risk_dir()
    print(f'Reading files from {risk_dir}')
    corpus = get_corpus(risk_dir)
    print(f'Read {len(corpus)} files.')

    model = Top2Vec(corpus, speed='deep-learn', workers=16)
    model_path = os.path.join(Config.top2vec_models_dir(),
                              'top2vec_model_deep')
    model.save(model_path)
    print(f'Saved model to {model_path}')
