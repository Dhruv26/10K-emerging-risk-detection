import glob
import os

# TODO: Update requirements
from nltk.tokenize import word_tokenize
from top2vec import Top2Vec
from tqdm import tqdm

from config import Config

if __name__ == '__main__':
    risk_dir = os.path.join(Config.risk_dir(), '10k20f_5')
    print(f'Reading files from {risk_dir}')
    risk_files = glob.glob(risk_dir)

    corpus = []
    for risk_file in tqdm(risk_files):
        docu = risk_file.read()
        if len(word_tokenize(docu)) > 100:
            corpus.append(docu)
    print(f'Read {len(corpus)} files.')

    model = Top2Vec(corpus, speed='deep-learn', workers=16)
    model_path = os.path.join(Config.models_dir(), 'top2vec_model')
    model.save(model_path)
    print(f'Saved model to {model_path}')
