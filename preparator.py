def remove_outliers(dataset):
    lower_bound = .25
    upper_bound = .75
    quant_df = dataset.quantile([lower_bound, upper_bound])

    filtering_rule_2 = dataset.apply(
        lambda x: (x < quant_df.loc[lower_bound, x.name]) | (x > quant_df.loc[upper_bound, x.name]), axis=0)

    dataframe = dataset[~(filtering_rule_2).any(axis=1)]
    return dataframe


def exponential_smoothing():
    pass


def dataframe_to_csv():
    pass


def csv_to_dataframe():
    pass
