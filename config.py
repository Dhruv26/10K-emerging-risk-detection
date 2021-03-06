import os


class Config:
    _project_path = os.path.dirname(os.path.abspath(__file__))
    _default_data_path = os.path.join(_project_path, 'data')
    _default_raw_report_zip_file = 'project_dataset_full.zip'

    @staticmethod
    def data_dir():
        return os.getenv('DATA_PATH', Config._default_data_path)

    @staticmethod
    def raw_report_zip_file():
        return os.getenv('RAW_REPORT_ZIP_FILE', os.path.join(
            Config.data_dir(), Config._default_raw_report_zip_file
        ))

    @staticmethod
    def risk_dir():
        return os.path.join(Config.data_dir(), 'risk_section')

    @staticmethod
    def log_dir():
        return os.getenv(
            'LOG_PATH', os.path.join(Config._project_path, 'logs')
        )

    @staticmethod
    def models_dir():
        return os.path.join(Config._project_path, 'models')

    @staticmethod
    def top2vec_models_dir():
        return os.path.join(Config.models_dir(), 'top2vec_models')

    @staticmethod
    def keywords_dir():
        return os.path.join(Config.models_dir(), 'keywords')

    @staticmethod
    def rake_keywords_dir():
        return os.path.join(Config.keywords_dir(), 'rake')

    @staticmethod
    def text_rank_keywords_dir():
        return os.path.join(Config.keywords_dir(), 'text_rank')

    @staticmethod
    def risk_sentiment_dir():
        return os.path.join(Config.models_dir(), 'risk_section_sentiment')

    @staticmethod
    def finBERT_model_dir():
        return os.path.join(Config.models_dir(), 'finBERT_model')

    @staticmethod
    def cache_dir():
        return os.path.join(Config._default_data_path, '.cache')

    @staticmethod
    def spacy_model():
        return os.getenv('SPACY_MODEL', 'en_core_web_md')
