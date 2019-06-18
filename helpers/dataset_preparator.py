import datetime

from helpers.preparator import remove_duplicates, delete_outliers, cut_dataframe, interpolate_nan, \
    cut_dataframe_by_period, get_data, sdd_missing_dates
from helpers.saver import df_to_csv


def prepare_dataframe(df, start=None, end=None, period_hours=None, sigma=2):
    if not end:
        end = start + datetime.timedelta(hours=period_hours)
    else:
        raise Exception('There should be end or period_hours')
    df = remove_duplicates(df)
    df = sdd_missing_dates(df)
    df = delete_outliers(df, m=sigma)
    df = cut_dataframe(df, start, end)
    df = interpolate_nan(df)
    return df


def prepare_csv(file, out_file, start=None, end=None, period_hours=None, sigma=2):
    df = get_data(file)
    df = prepare_dataframe(df, start=start, end=end, period_hours=period_hours, sigma=sigma)
    df.to_csv(out_file, index=True, encoding='utf-8-sig')
    return df
