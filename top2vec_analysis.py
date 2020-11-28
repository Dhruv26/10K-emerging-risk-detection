import glob
import os

# TODO: Update requirements
from nltk.tokenize import word_tokenize
from top2vec import Top2Vec

from config import Config

if __name__ == '__main__':
    risk_files = glob.glob(os.path.join(Config.risk_dir(), '10k20f_5'))
    corpus = []
    for risk_file in risk_files:
        docu = risk_file.read()
        if len(word_tokenize(docu)) > 100:
            corpus.append(docu)

    model = Top2Vec(corpus, speed='deep-learn', workers=16)
    model.save(os.path.join(Config.models_dir(), 'top2vec_model'))
