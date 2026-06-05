import pandas as pd

def reposition_column(dataframe, col_name, col_pos):
    col2repos = dataframe.pop(col_name)
    dataframe.insert(col_pos, col_name, col2repos.values)
    return dataframe

def daily_to_weekly_transform(daily_price_df) -> pd.DataFrame:
    """
    Aggregates a daily price DataFrame into a weekly OHLCV format.

    Parameters:
    -----------
    daily_price_df : pd.DataFrame
        Columns:
        ['symbol', 'date', 'open', 'high', 'low', 'close',
         'prev_close', 'volume', 'turnover cr', ...]
        'date' must be datetime64[ns].

    Returns:
    --------
    weekly_df : pd.DataFrame
        One row per symbol per trading week with:
        - date: first trading day of the week
        - open: first open
        - high: weekly high
        - low : weekly low
        - close: last close
        - prev_close: prev close of first trading day
        - volume / turnover / deliveries: summed
    """

    daily_df = daily_price_df.copy()
    daily_df = daily_df.sort_values(['symbol', 'date'])

    # ISO year-week avoids Dec/Jan bugs
    iso = daily_df['date'].dt.isocalendar()
    daily_df['iso_year'] = iso.year
    daily_df['iso_week'] = iso.week

    # First trading day of each symbol-week
    first_of_week = (
        daily_df
        .groupby(['symbol', 'iso_year', 'iso_week'], as_index=False)[['date']]
        .min()
        .rename(columns={'date': 'week_start_date'})
    )

    agg_dict = {
        'date': 'min',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'prev_close': 'first',
        'volume': 'sum',
    }

    # Optional fields
    opt_fields = [
        'num trades', 'turnover', 'turnover cr',
        'delqty', 'delval',
    ]
    aggfuncs = [
        'sum', 'sum', 'sum',
        'sum', 'sum',
    ]

    for field, aggfunc in zip(opt_fields, aggfuncs):
        if field in daily_df.columns:
            agg_dict.update({field: aggfunc})

    weekly_df = (
        daily_df
        .groupby(['symbol', 'iso_year', 'iso_week'], as_index=False)
        .agg(agg_dict)
        .sort_values(['symbol', 'date'])
        .reset_index(drop=True)
    )

    # Replace date with true first trading day
    weekly_df = (
        weekly_df
        .drop(columns='date')
        .merge(first_of_week,
               on=['symbol', 'iso_year', 'iso_week'],
               how='left')
        .rename(columns={'week_start_date': 'date'})
        .drop(columns=['iso_year', 'iso_week'])
    )

    weekly_df = reposition_column(weekly_df, 'date', 1)

    return weekly_df


def daily_to_monthly_transform(daily_price_df) -> pd.DataFrame:
    """
    Aggregates daily OHLCV data into monthly OHLCV.
    """

    daily_df = daily_price_df.copy()
    daily_df = daily_df.sort_values(['symbol', 'date'])

    # Use proper monthly period
    daily_df['month'] = daily_df['date'].dt.to_period('M')

    # First trading day per symbol-month
    first_of_month = (
        daily_df
        .groupby(['symbol', 'month'], as_index=False)[['date']]
        .min()
        .rename(columns={'date': 'month_start_date'})
    )

    agg_dict = {
        'date': 'min',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'prev_close': 'first',
        'volume': 'sum',
    }

    # Optional fields
    opt_fields = [
        'num trades', 'turnover', 'turnover cr',
        'delqty', 'delval',
    ]
    aggfuncs = [
        'sum', 'sum', 'sum',
        'sum', 'sum',
    ]

    for field, aggfunc in zip(opt_fields, aggfuncs):
        if field in daily_df.columns:
            agg_dict.update({field: aggfunc})

    monthly_df = (
        daily_df
        .groupby(['symbol', 'month'], as_index=False)
        .agg(agg_dict)
        .sort_values(['symbol', 'date'])
        .reset_index(drop=True)
    )

    # Replace date with true first trading day
    monthly_df = (
        monthly_df
        .drop(columns='date')
        .merge(first_of_month,
               on=['symbol', 'month'],
               how='left')
        .rename(columns={'month_start_date': 'date'})
        .drop(columns='month')
    )

    monthly_df = reposition_column(monthly_df, 'date', 1)

    return monthly_df
