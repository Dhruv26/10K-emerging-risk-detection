import os


class Config:
    _project_path = os.path.dirname(os.path.abspath(__file__))
    _default_data_path = os.path.join(_project_path, 'data')
    _default_raw_report_zip_file = 'project_dataset_full.zip'

    @staticmethod
    def data_path():
        return os.getenv('DATA_PATH', Config._default_data_path)

    @staticmethod
    def raw_report_zip_file():
        return os.getenv(
            'RAW_REPORT_ZIP_FILE',
            Config._default_raw_report_zip_file
        )

    @staticmethod
    def risk_dir():
        return os.path.join(Config._project_path, 'risk_section')
