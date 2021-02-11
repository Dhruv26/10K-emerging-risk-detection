from setuptools import setup, find_packages

setup(
    name='10K-emerging-risk-detection',
    version='0.1.0',
    packages=find_packages(include=('risk_detection',)),
    install_requires=[
        'beautifulsoup4==4.9.3',
        'lxml==4.6.1',
        'html2text==2020.1.16',
        'pandas>=1.1.4',
        'matplotlib==3.3.3',
        'scikit-learn==0.23.2',
        'torch==1.7.0+cpu',
        'torchaudio==0.7.0',
        'torchvision==0.8.1+cpu',
        'top2vec==1.0.16',
        'wordcloud==1.8.1',
        'tqdm==4.52.0',
        'keybert==0.1.3',
        'rake-nltk==1.0.4',
        'gensim==3.8.3',
        'spacy==2.2.4',
        'nltk==3.5',
        'finBert @ git+https://github.com/Dhruv26/finBERT.git@22dcc9e7417b04b30db49ef773110d50a600b936',
    ],
    setup_requires=['pytest-runner', 'flake8'],
    include_package_date=True,
)
