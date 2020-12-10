import os
import re
from datetime import datetime
from unicodedata import normalize

from bs4 import BeautifulSoup
from html2text import HTML2Text

from config import Config


class ReportInfo:
    def __init__(self, cik, start_date, end_date, filing_type, filename):
        self.cik = cik
        self.start_date = start_date
        self.end_date = end_date
        self.filing_type = filing_type
        self.filename = filename

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
        def _date_parser(date_str):
            return datetime.strptime(date_str, '%Y-%m-%d')

        cik, start_date, end_date, filing_type, filename = filename.split('_')
        return cls(int(cik), _date_parser(start_date), _date_parser(end_date),
                   filing_type, filename)


def extract_risk_section_from_report(raw_10k):
    """
    Extracts risk section from 10K SEC reports.

    :param raw_10k: The raw 10K report
    :return: Extracted risk section
    :raises: RiskSectionNotFound error if it cannot
             find the risk section in the report
    """
    doc_10k = _get_risk_document(raw_10k)
    raw_report = str(doc_10k)

    handler = _configure_html2text()

    report_text = handler.handle(raw_report)
    report_text_parts = report_text.splitlines()

    rf_section = _find_risk_factors_section(report_text_parts)
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
    for aapl_file in [r'data\aapl\0001193125-08-224958.txt']:
        try:
            print(f'Parsing {aapl_file}.')
            with open(aapl_file, 'r', encoding='utf-8') as f:
                doc = f.read()
            ex = extract_risk_section_from_report(doc)
            _write_risk_section(aapl_file, ex)
        except UnicodeDecodeError as e:
            print(f'Cannot open file {aapl_file}. {e}')
        except RiskSectionNotFound as e:
            print(f'Cannot extract risk section from file {aapl_file}. {e}')
