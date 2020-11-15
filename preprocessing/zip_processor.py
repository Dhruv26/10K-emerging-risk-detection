import os
from pathlib import Path
from zipfile import ZipFile
import time
import logging

from report_parser import extract_risk_section_from_report

LOGGER = logging.getLogger(__name__)


def write_risk_section_to_file(zipfile_path, output_dir):
    # Create output dir if it does not exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with ZipFile(zipfile_path) as zip_file:
        for report_file in zip_file.infolist():
            if report_file.is_dir():
                continue
            report = zip_file.read(report_file)
            try:
                risk_section = extract_risk_section_from_report(report)
            except:
                error_msg = f'Error occurred while trying to extract risk ' \
                            f'section for {report_file.filename} '
                LOGGER.error(error_msg, exc_info=True)
                continue

            output_file = os.path.join(output_dir, report_file.filename)
            with open(output_file, 'w+') as risk_section_file:
                risk_section_file.write(risk_section)
            LOGGER.info(f'Extracted risk section for {report_file.filename}')


if __name__ == '__main__':
    # TODO: Configure logging at project level
    logging_format = '%(asctime)s — %(name)s — %(levelname)s — %(funcName)s' \
                     ':%(lineno)d — %(message)s'
    logging.basicConfig(
        filename='risk_extractor.log',
        format=logging_format
    )

    input_zipfile = 'project_dataset_full.zip'
    output_dir = 'risk_section'
    LOGGER.info(f'Processing {input_zipfile}')

    start = time.time()
    write_risk_section_to_file(input_zipfile, output_dir)
    time_taken = time.time() - start

    LOGGER.info(
        f'Written result to {output_dir}. Took {time_taken:.2f} seconds.'
    )
