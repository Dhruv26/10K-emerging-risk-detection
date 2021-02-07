import os
from typing import Dict, Tuple

import pandas as pd
from pkg_resources import resource_stream

sheet_names = [
    'Div. A - Agr., Forstry, Fishing', 'Div. B - Mining',
    'Div. C - Construction Industry', 'Div. D - Manufacturing',
    'Div. E -Transport., Com., Util.', 'Div. F - Wholesale Trade',
    'Div. G - Retail Trade', 'Div. H - Fin., Ins., RE.',
    'Div. I - Service Industries', 'Div. J - Public Administration'
]


def _clean_sheet_df(sheet_df: pd.DataFrame) -> pd.DataFrame:
    sheet_df = sheet_df.drop('X', axis=1).dropna(axis=0, how='all')
    cols = ['MAJOR GROUP', 'INDUSTRY GROUP', 'MOST SPECIFIC SIC CODE']
    for col in cols:
        sheet_df.loc[1:, col] = sheet_df.loc[1:, col].str[1:]
    return sheet_df


def _get_sic_industry_name_file() -> Tuple[Dict[str, pd.DataFrame], str]:
    industry_name_file = resource_stream(
        'preprocessing', os.path.join('resources', 'sic_industry_name.xlsx')
    )
    return (pd.read_excel(
        industry_name_file, engine='openpyxl', sheet_name=sheet_names
    ),
            os.path.dirname(industry_name_file.name)
    )


if __name__ == '__main__':
    print('Reading input file.')
    sheets, file_dir = _get_sic_industry_name_file()
    cleaned_dfs = [_clean_sheet_df(sheet_df) for sheet_df in sheets.values()]
    df = pd.concat(cleaned_dfs, ignore_index=True)

    csv_file_name = os.path.join(file_dir, 'sic_industry_name.csv')
    df.to_csv(csv_file_name, index=False)
    print(f'Written cleaned file to {csv_file_name}')
