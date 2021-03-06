import atexit
import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path
from zipfile import ZipFile

from config import Config
from report_parser import (extract_risk_section_from_report, ReportInfo,
                           RiskSectionNotFound)


def _configure_log_process():
    queue = multiprocessing.Manager().Queue(-1)
    file_handler = logging.FileHandler(
        os.path.join(Config.log_dir(), 'risk_extractor_multiprocessing.log')
    )
    formatter = '%(asctime)s-[%(levelname)s]-%(name)s-%(filename)s.' \
                '%(funcName)s(%(lineno)d)-%(message)s'
    file_handler.setFormatter(logging.Formatter(formatter))
    queue_listener = QueueListener(queue, file_handler)
    queue_listener.start()
    return queue, queue_listener


_queue, _queue_listener = _configure_log_process()
_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(QueueHandler(_queue))


@atexit.register
def _on_exit():
    _queue_listener.stop()


def list_of_files_to_extract_from_zip(zipfile_path):
    with ZipFile(zipfile_path) as zip_file:
        report_files = zip_file.infolist()
    return map(
        lambda zip_fileinfo: zip_fileinfo.filename,
        filter(lambda file_in_zip: not file_in_zip.is_dir(), report_files)
    )


def process_zipfile(zipfile_path, report_file, output_dir):
    with ZipFile(zipfile_path) as zip_file:
        report = zip_file.read(report_file)

    file_dir, filename = os.path.split(report_file)
    report_info = ReportInfo.from_zip_filename(filename)
    try:
        risk_section = extract_risk_section_from_report(report)
        _write_risk_section_to_file(report_info, risk_section, output_dir)
        _LOGGER.info(
            f'Extracted risk section for {report_file}'
        )
    except RiskSectionNotFound:
        error_msg = f'Error occurred while trying to extract risk ' \
                    f'section for {report_file}'
        _LOGGER.error(error_msg, exc_info=True)
    except RecursionError:
        # Hate that python does not optimize tail recursion and bs4
        # uses a recursive structure and algos to parse the docs.
        # Or it's just a vv badly formatted file!
        # 799292_2010-12-31_2011-03-01_10-K_0000799292-11-000010.txt
        _LOGGER.error(f'Check the contents of {report_file}.')


def _write_risk_section_to_file(report_info, risk_section, output_dir):
    output_file = os.path.join(output_dir, str(report_info.cik),
                               report_info.get_file_name())
    output_file_dir = os.path.dirname(output_file)
    Path(output_file_dir).mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w+') as risk_section_file:
        risk_section_file.write(risk_section)


def _process_zipfile_wrapper(args):
    process_zipfile(args[0], args[1], args[2])


def main():
    input_zipfile_path = Config.raw_report_zip_file()
    output_dir = Config.risk_dir()

    _LOGGER.info(f'Processing {input_zipfile_path}')

    start = time.time()
    zip_files = list_of_files_to_extract_from_zip(input_zipfile_path)
    args = [
        (input_zipfile_path, zip_file, output_dir) for zip_file in zip_files
    ]
    with ProcessPoolExecutor(max_workers=10) as executor:
        executor.map(_process_zipfile_wrapper, args, chunksize=20)
    time_taken = time.time() - start

    _LOGGER.info(
        f'Written result to {output_dir}. Took {time_taken:.2f} seconds.'
    )


if __name__ == '__main__':
    main()
