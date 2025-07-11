import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from itertools import combinations

def rank_interactions_by_correlation(interaction_df, target_series):
    """
    Compute and rank the correlation of interaction features with the target variable.

    Parameters:
    interaction_df (pd.DataFrame): DataFrame containing interaction features.
    target_series (pd.Series): Target variable to correlate with.

    Returns:
    pd.DataFrame: Sorted DataFrame with features and their correlation to the target.
    """
    correlations = interaction_df.corrwith(target_series)
    correlation_df = correlations.reset_index()
    correlation_df.columns = ['Feature', 'Correlation']
    correlation_df = correlation_df.sort_values(by='Correlation', key=abs, ascending=False).reset_index(drop=True)
    return correlation_df

def create_2nd_degree_interactions(df):
    """
    Generate all second-degree interaction-only combinations of numeric columns
    in the format 'column1 x column2'.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: A DataFrame of interaction features.
    """
    numeric_cols = df.select_dtypes(include='number').columns
    interactions = {}

    for col1, col2 in combinations(numeric_cols, 2):
        interaction_name = f"{col1} x {col2}"
        interactions[interaction_name] = df[col1] * df[col2]

    return pd.DataFrame(interactions, index=df.index)


def calculate_dryness_index(df):
    # Work on a copy of the input DataFrame
    df_copy = df.copy()
    
    # Define features and scale them (if needed elsewhere)
    features = ['Avg Air Temp (F)', 'Avg Wind Speed (mph)', 'Sol Rad (Ly/day)', 'Avg Rel Hum (%)','Precip (in)']
    
    scaler = StandardScaler()
    _ = scaler.fit_transform(df_copy[features])  # optional, not used in this function

    # Calculate PET proxy and Dryness on the copy
    df['PET_proxy'] = (
        df_copy['Avg Air Temp (F)'] +
        df_copy['Avg Wind Speed (mph)'] +
        df_copy['Sol Rad (Ly/day)'] -
        df_copy['Avg Rel Hum (%)']
    )

    df['Dryness'] = df['Precip (in)'] - df['PET_proxy']

    return df['Dryness']
    
def impute_median_data(dataframe):
    """
    Impute missing numeric values using column medians.

    Parameters:
        dataframe (pd.DataFrame): DataFrame with numeric columns to impute.

    Returns:
        pd.DataFrame: A new DataFrame with NaNs replaced by medians.
    """
    df = dataframe.copy()
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    return df

def data_explore(df):
    print('Duplicates before dropping: ', df.duplicated().sum(), '\n')
    df = df.drop_duplicates()
    df.info()
    return df

def post_merge_check (merged_df, premerged_df):
    
    print("Duplicate Keys: ", merged_df[['Date','Stn Id']].duplicated().sum()) 
    print("Premerged shape: ", premerged_df.shape)
    print("Merged shape: ", merged_df.shape)
    print("Duplicates after merge: ", merged_df.duplicated().sum())
    print('NA values before merge: ', premerged_df.isna().sum().sum())    
    print('NA values after merge: ', merged_df.isna().sum().sum())    

def identify_missing_station_dates(weather_df, start='2018-01-01', end='2020-09-18'):
    """
    Identifies missing daily weather records for each station (`Stn Id`) 
    based on an expected date range.

    Returns:
        pd.DataFrame: A DataFrame containing the Stn Id and missing Date for each gap found.
    """
    expected_dates = pd.date_range(start=start, end=end, freq='D')
    missing_records = []

    station_ids = weather_df['Stn Id'].unique()

    for stn_id in station_ids:
        # Filter rows for the current station
        station_data = weather_df[weather_df['Stn Id'] == stn_id]
        station_dates = pd.to_datetime(station_data['Date']).dt.date

        for date in expected_dates:
            if date.date() not in station_dates.values:
                missing_records.append({
                    'Stn Id': stn_id,
                    'Date': date.date()
                })

    return pd.DataFrame(missing_records)

########################################################################################
#### FUNCTIONS BELOW ARE CURRENTLY UNUSED BUT MAY BE IMPLEMENTED IN THE FUTURE #########
########################################################################################

def identify_missing_dates(weather_df, start='2018-01-01', end='2020-09-18'):
    """
    Identifies missing daily weather records for each county based on an expected date range.

    Returns:
        pd.DataFrame: A dataframe containing the Date, County, Elevation, and CIMIS Region 
                      for each missing observation.
    """
    # Get a list of all unique counties from the dataset
    counties = weather_df['County'].unique()

    # Create empty lists to store information about missing entries
    missing_dates = []
    missing_countys = []
    missing_stations = []
    missing_elevations = []
    missing_regions = []

    # Loop through each county to find missing dates
    for county in counties:
        # Create a working copy of the data for the current county
        df_county = weather_df[weather_df['County'] == county].copy()

        # Reset index to make row-wise operations easier
        df_county = df_county.reset_index(drop=True)

        # Create a complete list of dates for the desired time range
        dates = pd.date_range(start=start, end=end, freq='D')

        # If the current county dataset has fewer entries than expected
        if len(dates) > len(df_county):
            # Calculate how many dates are missing
            length_missing = len(dates) - len(df_county)

            # Create a DataFrame of empty rows (NaNs) to pad the data
            missing_rows = pd.DataFrame(np.nan, index=range(length_missing), columns=df_county.columns)

            # Append the missing rows to match the expected length
            df_county = pd.concat([df_county, missing_rows], ignore_index=True)

        # Convert the 'Date' column to datetime for comparison
        county_date = pd.to_datetime(df_county['Date'])

        # Loop through each expected date to identify any that are missing
        for i in range(len(dates)):
            date = pd.to_datetime(dates[i]).date()

            # If the expected date is not found in the data
            if date not in county_date.dt.date.values:
                # Record the missing date
                missing_dates.append(date)

                # Find the first non-null row to extract metadata (County, Elevation, Region)
                j = 0
                while pd.isna(df_county.iloc[j]['County']):
                    j += 1

                # Append the county, elevation, and CIMIS region associated with the missing date
                missing_countys.append(df_county.iloc[j]['County'])
                missing_elevations.append(df_county.iloc[j]['Elevation'])
                missing_regions.append(df_county.iloc[j]['CIMIS Region'])

    # Construct and return the DataFrame
    missing_records = pd.DataFrame({
        'Date': missing_dates,
        'County': missing_countys,
        'Elevation': missing_elevations,
        'CIMIS Region': missing_regions
    })

    return missing_records



def impute_missing_rows(weather_df, missing_df):
    """
    Fills in missing weather records using the mean of other stations in the same region
    on the same date. Skips rows if no data exists for that region/date.
    """
    weather_df['Date'] = pd.to_datetime(weather_df['Date'])
    columns = weather_df.columns

    for _, row in missing_df.iterrows():
        target_date = pd.to_datetime(row['Date'])

        # Filter data for same region and same date
        region_data = weather_df[
            (weather_df['CIMIS Region'] == row['CIMIS Region']) &
            (weather_df['Date'] == target_date)
        ]

        if region_data.empty:
            continue  # Skip if no data available for that region on that date

        numeric_columns = region_data.select_dtypes(include='number').columns
        new_row = {col: np.nan for col in columns}
        new_row['Date'] = target_date
        new_row['County'] = row['County']
        new_row['Elevation'] = row['Elevation']
        new_row['CIMIS Region'] = row['CIMIS Region']

        for col in numeric_columns:
            new_row[col] = region_data[col].mean(skipna=True)

        weather_df = pd.concat([weather_df, pd.DataFrame([new_row])], ignore_index=True)

    return weather_df
