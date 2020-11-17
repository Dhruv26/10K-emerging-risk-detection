import logging
import os
import time
from pathlib import Path
from zipfile import ZipFile

from report_parser import extract_risk_section_from_report, RiskSectionNotFound

from config import Config

LOGGER = logging.getLogger(__name__)


def write_risk_section_to_file(zipfile_path, output_dir):
    # Create output dir if it does not exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with ZipFile(zipfile_path) as zip_file:
        for report_file in zip_file.infolist():
            if report_file.is_dir():
                continue

            LOGGER.info(f'Processing {report_file.filename}')
            report = zip_file.read(report_file)
            try:
                risk_section = extract_risk_section_from_report(report)
                output_file = os.path.join(output_dir, report_file.filename)
                output_file_dir = os.path.dirname(output_file)
                Path(output_file_dir).mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w+') as risk_section_file:
                    risk_section_file.write(risk_section)
                LOGGER.info(
                    f'Extracted risk section for {report_file.filename}'
                )
            except RiskSectionNotFound:
                error_msg = f'Error occurred while trying to extract risk ' \
                            f'section for {report_file.filename} '
                LOGGER.error(error_msg, exc_info=True)
                continue


if __name__ == '__main__':
    logging_format = '%(asctime)s — %(levelname)s — %(funcName)s:%(' \
                     'lineno)d — %(message)s'
    logging.basicConfig(
        filename=os.path.join(Config.log_dir(), 'risk_extractor.log'),
        format=logging_format,
        level=logging.DEBUG
    )

    input_zipfile_path = os.path.join(
        Config.data_path(),
        Config.raw_report_zip_file()
    )
    output_dir = Config.risk_dir()
    LOGGER.info(f'Processing {input_zipfile_path}')

    start = time.time()
    write_risk_section_to_file(input_zipfile_path, output_dir)
    time_taken = time.time() - start

    LOGGER.info(
        f'Written result to {output_dir}. Took {time_taken:.2f} seconds.'
    )
