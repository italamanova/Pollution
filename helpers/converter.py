from datetime import datetime


def str_to_datetime(str_datetime):
    return datetime.strptime(str_datetime, '%Y-%m-%d %H:%M:%S')


def date_str_to_datetime(str_date):
    return datetime.strptime('%s 00:00:00' % str_date, '%Y-%m-%d %H:%M:%S')
