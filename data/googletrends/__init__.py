import os
import numpy as np
import pandas as pd


def load_data(query, geo_level, sampling_rate, start_date=None, end_date=None):
    data = pd.read_csv(os.path.dirname(__file__)
        + f'/{query}_{geo_level}_{sampling_rate}.csv.gz')
    
    if start_date is not None:
        data.date = pd.to_datetime(data.date)
        data = data[lambda x: x.date >= start_date]
    if end_date is not None:
        data.date = pd.to_datetime(data.date)
        data = data[lambda x: x.date <= end_date]

    return data


def load_as_tensor(query, geo_level, sampling_rate,
                   start_date=None, end_date=None):
    
    print('query= {}; geo_level= {}; sampling_rate= {}'.format(
        query, geo_level, sampling_rate))

    # Load original data
    data = load_data(query, geo_level, sampling_rate)
    data['date'] = pd.to_datetime(data['date'])

    if start_date is not None:
        data = data[lambda x: x['date'] >= start_date]
    if end_date is not None:
        data = data[lambda x: x['date'] <= end_date]

    print("Duration:", data['date'].min(), data['date'].max())
    # DataFrame to Tensor Timeseries
    return df2tts(data, 'date', ['keyword', geo_level], values='value')


def df2tts(df, time_key, facets, values=None, sampling_rate="D"):
    """ Convert a DataFrame (list) to tensor time series

        df (pandas.DataFrame):
            A list of discrete events
        time_key (str):
            A column name of timestamps
        facets (list):
            A list of column names to make tensor timeseries
        values (str):
            A column name of target values (optional)
        sampling_rate (str):
            A frequancy for resampling, e.g., "7D", "12H", "H"
    """
    tmp = df.copy(deep=True)
    shape = tmp[facets].nunique().tolist()
    if values == None: values = 'count'; tmp[values] = 1
    tmp[time_key] = tmp[time_key].round(sampling_rate)
    print("Tensor:")
    print(tmp.nunique()[[time_key] + facets])

    grouped = tmp.groupby([time_key] + facets).sum()[[values]]
    grouped = grouped.unstack(fill_value=0).stack()
    grouped = grouped.pivot_table(index=time_key, columns=facets, values=values, fill_value=0)

    tts = grouped.values
    tts = np.reshape(tts, (-1, *shape))
    return tts

