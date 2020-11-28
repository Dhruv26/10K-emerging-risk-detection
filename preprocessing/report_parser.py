import json
import os
import re
from datetime import datetime
from io import BytesIO
from unicodedata import normalize

import pandas as pd
from bs4 import BeautifulSoup
from pkg_resources import resource_string

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
    def from_zip_filename(cls, zip_filename):
        def _date_parser(date_str):
            return datetime.strptime(date_str, '%Y-%m-%d')

        zip_dir, filename = os.path.split(zip_filename)
        cik, start_date, end_date, filing_type, filename = filename.split('_')
        return cls(int(cik), _date_parser(start_date), _date_parser(end_date),
                   filing_type, filename)

    @classmethod
    def from_risk_factors_file(cls, row):
        return cls(row['cik'], row['periodofreport'].to_pydatetime(),
                   row['filingdate'].to_pydatetime(), row['documenttype'],
                   row['documentlinktext'])


def _read_risk_start_end_data():
    risk_start_end_csv = resource_string(
        'preprocessing', os.path.join('resources', 'RiskFactors_StartEnd.csv')
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


def extract_risk_section_from_report(raw_10k):
    """
    Extracts risk section from 10K SEC reports.

    :param raw_10k: The raw 10K report
    :return: Extracted risk section
    :raises: RiskSectionNotFound error if it cannot
             find the risk section in the report
    """
    types = _get_risk_document(raw_10k)

    raw_report = str(types)
    pos_data = _get_risk_section_from_soup(raw_report)
    raw_risk = raw_report[
               pos_data['start'].loc['item1a']:pos_data['start'].loc['item1b']
               ]
    risk_soup = BeautifulSoup(raw_risk, 'lxml')

    # The first p tag contains the Risk Section title. Hence we remove it.
    risk_title_tag = risk_soup.find('p')
    if risk_title_tag:
        risk_title_tag.decompose()
    return re.sub(r'(\n|\s)+', ' ', _restore_windows_1252_characters(
        normalize("NFKD", risk_soup.get_text())
    )).strip()


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


def _get_risk_section_from_soup(raw_report):
    # Regex to find start of section 1A and 1B
    risk_section_regex = re.compile(
        r'(>Item(\s|&#160;|&nbsp;)+(1A|1B)\.?)|(ITEM(\s)+(1A|1B))'
    )
    risk_section_matches = [
        {
            'item': x.group(), 'start': x.start(), 'end': x.end()
        }
        for x in (risk_section_regex.finditer(raw_report))
    ]
    if not risk_section_matches:
        raise RiskSectionNotFound('Risk section not found.')

    match_df = pd.read_json(json.dumps(risk_section_matches))
    match_df['item'] = match_df.item.str.lower()
    return _create_matches_df(match_df)


def _create_matches_df(match_df):
    # Get rid of unnecessary characters from the dataframe
    match_df.replace('&#160;', ' ', regex=True, inplace=True)
    match_df.replace('&nbsp;', ' ', regex=True, inplace=True)
    match_df.replace(' ', '', regex=True, inplace=True)
    match_df.replace('\.', '', regex=True, inplace=True)
    match_df.replace('>', '', regex=True, inplace=True)

    _validate_match_df(match_df)
    return (match_df.sort_values('start', ascending=True)
            .drop_duplicates(subset=['item'], keep='last')
            .set_index('item'))


def _validate_match_df(match_df):
    unique_items = match_df['item'].unique()
    if 'item1a' not in unique_items:
        raise RiskSectionNotFound('Item 1A not found.')
    if 'item1b' not in unique_items:
        raise RiskSectionNotFound('Item 1B not found.')


class RiskSectionNotFound(Exception):
    def __init__(self, message):
        super().__init__(message)


if __name__ == '__main__':
    filename = r'10k20f_10\798949_2010-12-31_2011-02-24_10-K_0001193125-11-044656.txt'
    with open(os.path.join(Config.data_dir(), filename), 'r') as f:
        doc = f.read()

    import glob
    aapl_files = glob.glob(os.path.join(Config.data_dir(), 'aapl', '*.txt'))
    for aapl_file in aapl_files:
        try:
            print(f'Parsing {aapl_file}.')
            with open(aapl_file, 'r') as f:
                doc = f.read()
            ex = extract_risk_section_from_report(doc)
            with open(
                    os.path.join(Config.risk_dir(), aapl_file.split('\\')[-1]),
                    'w+',
                    encoding='cp1252'
            ) as f:
                f.write(ex)
                # import pdb; pdb.set_trace()
        except (UnicodeDecodeError, RiskSectionNotFound):
            continue

    report_info = ReportInfo.from_zip_filename(filename)
