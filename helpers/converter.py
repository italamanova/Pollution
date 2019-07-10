from datetime import datetime


def get_resampled(df, period_name):
    resampled = df.resample(period_name).sum()
    return resampled


def str_to_datetime(str_datetime):
    return datetime.strptime(str_datetime, '%Y-%m-%d %H:%M:%S')


def date_str_to_datetime(str_date):
    return datetime.strptime('%s 00:00:00' % str_date, '%Y-%m-%d %H:%M:%S')


def datetime_to_string(datetime):
    if isinstance(datetime, (datetime.date, datetime.datetime)):
        return datetime.isoformat()


def df_to_csv(df, out_file):
    df.to_csv(out_file, encoding='utf-8-sig')
