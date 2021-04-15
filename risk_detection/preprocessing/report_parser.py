import os
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import List
from unicodedata import normalize

import pandas as pd
from bs4 import BeautifulSoup
from html2text import HTML2Text
from pkg_resources import resource_string

from config import Config


class ReportInfo:
    DEFAULT_FILING_TYPE = '10-K'

    def __init__(self, cik: int, start_date: datetime, end_date: datetime,
                 filing_type: str, filename: str):
        self.cik = cik
        self.start_date = start_date
        self.end_date = end_date
        self.filing_type = filing_type
        self.filename = filename

    def get_file_name(self) -> str:
        return f'{self.start_date.date()}_{self.end_date.date()}_' \
               f'{self.filename}'

    def get_document_id(self) -> str:
        return f'{self.cik}_{self.start_date.date()}_{self.end_date.date()}_' \
               f'{self.filename}'

    def _keys(self):
        return (self.cik, self.start_date, self.end_date, self.filing_type,
                self.filename)

    def __eq__(self, other):
        if isinstance(other, ReportInfo):
            return self._keys() == other._keys()
        return False

    def __hash__(self):
        return hash(self._keys())

    def __repr__(self):
        return (f'CIK: {self.cik}, Start Date: {self.start_date}, '
                f'End Date: {self.end_date}, Filing Type: {self.filing_type}, '
                f'File Name: {self.filename}')

    @classmethod
    def from_zip_filename(cls, filename):
        cik, start_date, end_date, filing_type, filename = filename.split('_')
        return cls(int(cik), _date_parser(start_date), _date_parser(end_date),
                   filing_type, filename)

    @classmethod
    def from_doc_id(cls, doc_id: str):
        cik, start_date, end_date, filename = doc_id.split('_', maxsplit=4)
        return cls(int(cik), _date_parser(start_date), _date_parser(end_date),
                   cls.DEFAULT_FILING_TYPE, filename)

    @classmethod
    def from_risk_factors_file(cls, row):
        return cls(row['cik'], row['periodofreport'].to_pydatetime(),
                   row['filingdate'].to_pydatetime(), row['documenttype'],
                   row['documentlinktext'])


def _date_parser(date_str: str) -> datetime:
    return datetime.strptime(date_str, '%Y-%m-%d')


def report_info_from_risk_path(risk_file: Path) -> ReportInfo:
    cik = int(risk_file.parent.name)
    file_name = risk_file.name
    start_date, end_date, filename = file_name.split('_', maxsplit=2)
    return ReportInfo(cik, _date_parser(start_date), _date_parser(end_date),
                      '10-K', filename)


def _read_risk_start_end_data():
    risk_start_end_csv = resource_string(
        'preprocessing', os.path.join('static', 'RiskFactors_StartEnd.csv')
    )
    date_cols = ['periodofreport', 'filingdate']
    cols = [
        'cik', 'documenttype', 'periodofreport', 'filingdate',
        'documentlinktext', 'sectionstart', 'sectionend'
    ]
    df = pd.read_csv(BytesIO(risk_start_end_csv), usecols=cols,
                     parse_dates=date_cols)
    df['documentlinktext'] = df['documentlinktext'].str.split('/').str[-1]
    df['report_file_info'] = df.apply(ReportInfo.from_risk_factors_file,
                                      axis=1)
    return (df.drop(['cik', 'documenttype', 'periodofreport', 'filingdate',
                     'documentlinktext'], axis=1)
            .set_index('report_file_info')
            .to_dict(orient='index'))


_risk_start_end_dict = _read_risk_start_end_data()


def extract_risk_section_from_report(raw_10k: str,
                                     report_info: ReportInfo) -> str:
    """
    Extracts risk section from 10K SEC reports.

    :param raw_10k: The raw 10K report
    :param report_info: Report Information
    :return: Extracted risk section
    :raises: RiskSectionNotFound error if it cannot
             find the risk section in the report
    """
    doc_10k = _get_risk_document(raw_10k)
    raw_report = str(doc_10k)

    handler = _configure_html2text()

    report_text = handler.handle(raw_report)
    report_text_parts = report_text.splitlines()

    try:
        rf_section = _find_risk_factors_section(report_text_parts)
    except RiskSectionNotFound:
        # Look for risk section start end in metadata dataframe
        rf_section = _find_in_metadata_file(report_info, report_text_parts)

    return ' '.join(rf_section)


def _configure_html2text():
    handler = HTML2Text()

    handler.ignore_emphasis = True
    handler.ignore_links = True
    handler.ignore_images = True
    handler.ignore_tables = True
    handler.bypass_tables = True

    return handler


def _find_risk_factors_section(report_text_parts):
    start_found = end_found = False
    rf_section = list()
    for idx, line in enumerate(report_text_parts):
        pattern = re.compile(r'[\W_]+|(Table of Contents)')
        line_clean = pattern.sub('', line)
        if (not start_found and
                re.match('item1ariskfactors', line_clean, re.I)):
            start_found = True
        elif re.match('item1bUnresolvedStaffComments', line_clean, re.I):
            end_found = True
            break
        elif start_found and line_clean and not line_clean.isdigit():
            rf_section.append(_restore_windows_1252_characters(
                normalize("NFKD", line)
            ))

    if not start_found or not end_found or not len(rf_section):
        raise RiskSectionNotFound(
            f'Item 1A found: {start_found}, Item 1B found: {end_found}, '
            f'Text found between them: {len(rf_section) != 0}'
        )
    return rf_section


def _find_in_metadata_file(report_info: ReportInfo,
                           report_text_parts: List[str]) -> str:
    pattern = re.compile(r'[\W_]+|(Table of Contents)')
    start_end_dict = _risk_start_end_dict[report_info]
    section_start = pattern.sub('', start_end_dict['sectionstart'])
    section_end = pattern.sub('', start_end_dict['sectionend'])

    start_found = end_found = False
    rf_section = list()
    for idx, line in enumerate(report_text_parts):
        line_clean = pattern.sub('', line)
        if (not start_found and
                re.match(section_start, line_clean, re.I)):
            start_found = True
        elif re.match(section_end, line_clean, re.I):
            end_found = True
            break
        elif start_found and line_clean and not line_clean.isdigit():
            rf_section.append(_restore_windows_1252_characters(
                normalize("NFKD", line)
            ))

    if not start_found or not end_found or not len(rf_section):
        raise RiskSectionNotFound(
            f'Item 1A found: {start_found}, Item 1B found: {end_found}, '
            f'Text found between them: {len(rf_section) != 0}'
        )
    return rf_section


def _restore_windows_1252_characters(restore_string):
    """
    Replace C1 control characters in the Unicode string s by the
    characters at the corresponding code points in Windows-1252,
    where possible.
    """

    def to_windows_1252(match):
        try:
            return bytes([ord(match.group(0))]).decode('windows-1252')
        except UnicodeDecodeError:
            # No character at the corresponding code point: remove it.
            return ''

    return re.sub(r'[\u0080-\u0099]', to_windows_1252, restore_string)


def _get_risk_document(raw_10k):
    soup = BeautifulSoup(raw_10k, 'lxml')
    return soup.find(
        lambda x: x.name == 'type' and x.get_text().startswith('10-K')
    )


class RiskSectionNotFound(Exception):
    def __init__(self, message):
        super().__init__(message)


if __name__ == '__main__':
    def _write_risk_section(aapl_file, ex):
        with open(
                os.path.join(Config.risk_dir(), aapl_file.split('\\')[-1]),
                'w+',
                encoding='cp1252'
        ) as f:
            f.write(ex)

    import glob
    aapl_files = glob.glob(os.path.join(Config.data_dir(), 'aapl', '*.txt'))
    for aapl_file in aapl_files:
        try:
            print(f'Parsing {aapl_file}.')
            with open(aapl_file, 'r', encoding='utf-8') as f:
                doc = f.read()
            ex = extract_risk_section_from_report(doc, None)
            _write_risk_section(aapl_file, ex)
        except UnicodeDecodeError as e:
            print(f'Cannot open file {aapl_file}. {e}')
        except RiskSectionNotFound as e:
            print(f'Cannot extract risk section from file {aapl_file}. {e}')
