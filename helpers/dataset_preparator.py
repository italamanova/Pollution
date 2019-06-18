from helpers.preparator import remove_duplicates, delete_outliers, cut_dataframe, interpolate_nan, \
    cut_dataframe_by_period, get_data
from helpers.saver import df_to_csv


def prepare_dataframe(df, start=None, end=None, period_hours=None, sigma=2):
    df = remove_duplicates(df)
    df = delete_outliers(df, m=sigma)
    if start and end:
        df = cut_dataframe(df, start, end)
    elif start and period_hours:
        df = cut_dataframe_by_period(df, start, period_hours)
    df = interpolate_nan(df)
    return df


def prepare_csv(file, out_file, start=None, end=None, period_hours=None, sigma=2):
    df = get_data(file)
    col_name = df.columns[0]
    df = prepare_dataframe(df, start=start, end=end, period_hours=period_hours, sigma=sigma)
    df.to_csv(out_file, columns=[col_name], index=True, encoding='utf-8-sig')
    return df
