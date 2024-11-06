import pandas as pd

def load_data(filepath, columns=None):
    """
    Loads a CSV file and filters for specific columns if provided.

    Parameters:
    - filepath (str): Path to the CSV file.
    - columns (list of str): List of column names to keep in the loaded data.

    Returns:
    - pd.DataFrame: The loaded DataFrame, filtered by specified columns.
    """
    data = pd.read_csv(filepath,skiprows=4)
    if columns:
        data = data[columns]
    return data

def filter_by_countries(data, countries):
    """
    Filters the data to include only specified countries.

    Parameters:
    - data (pd.DataFrame): The input DataFrame containing a 'Country Name' column.
    - countries (list of str): List of country names to filter for.

    Returns:
    - pd.DataFrame: The filtered DataFrame with only the specified countries.
    """
    return data[data['Country Name'].isin(countries)]

def reshape_to_long_format(data, id_vars, var_name='Year', value_name='Value'):
    """
    Reshapes the data from wide format to long format using pd.melt.

    Parameters:
    - data (pd.DataFrame): The input DataFrame in wide format.
    - id_vars (list of str): The columns to keep as identifiers.
    - var_name (str): Name of the new 'variable' column, default is 'Year'.
    - value_name (str): Name of the new 'value' column, default is 'Value'.

    Returns:
    - pd.DataFrame: The reshaped DataFrame in long format.
    """
    data_long = data.melt(id_vars=id_vars, var_name=var_name, value_name=value_name)
    data_long[var_name] = pd.to_numeric(data_long[var_name])
    return data_long

def reshape_to_long_format(data, id_vars, var_name='Year', value_name='Value'):
    """
    Reshapes the data from wide format to long format using pd.melt.

    Parameters:
    - data (pd.DataFrame): The input DataFrame in wide format.
    - id_vars (list of str): The columns to keep as identifiers.
    - var_name (str): Name of the new 'variable' column, default is 'Year'.
    - value_name (str): Name of the new 'value' column, default is 'Value'.

    Returns:
    - pd.DataFrame: The reshaped DataFrame in long format.
    """
    data_long = data.melt(id_vars=id_vars, var_name=var_name, value_name=value_name)
    data_long[var_name] = pd.to_numeric(data_long[var_name])
    return data_long

def filter_by_year(data, start_year, end_year):
    """
    Filters the data for rows within a specific year range.

    Parameters:
    - data (pd.DataFrame): The input DataFrame with a 'Year' column.
    - start_year (int): The starting year for the filter.
    - end_year (int): The ending year for the filter.

    Returns:
    - pd.DataFrame: The filtered DataFrame.
    """
    return data[(data['Year'] >= start_year) & (data['Year'] <= end_year)]

def resample_annual_avg(data, date_column, value_column):
    """
    Resamples a time series data to annual frequency by taking the mean for each year.

    Parameters:
    - data (pd.DataFrame): The input DataFrame with a datetime column.
    - date_column (str): The name of the column containing date information.
    - value_column (str): The name of the column containing the values to be averaged.

    Returns:
    - pd.DataFrame: The resampled DataFrame with annual averages.
    """
    data[date_column] = pd.to_datetime(data[date_column])
    data.set_index(date_column, inplace=True)
    annual_data = data[value_column].resample('Y').mean().reset_index()
    annual_data['Year'] = annual_data[date_column].dt.year
    return annual_data[['Year', value_column]]

def merge_datasets(economic_data, brent_data, on='Year', indicator_name=None):
    """
    Merges two datasets on a specified column and optionally filters by indicator.

    Parameters:
    - economic_data (pd.DataFrame): Economic data containing indicators and years.
    - brent_data (pd.DataFrame): Brent oil data with years and oil prices.
    - on (str): The column name to merge on, default is 'Year'.
    - indicator_name (str): If provided, filter economic data to only this indicator.

    Returns:
    - pd.DataFrame: The merged DataFrame.
    """
    if indicator_name:
        economic_data = economic_data[economic_data['Indicator Name'] == indicator_name]
    
    merged_data = pd.merge(economic_data, brent_data, on=on, how='inner')
    return merged_data

def handle_missing_values(data, method='ffill'):
    """
    Handles missing values in the dataset.

    Parameters:
    - data (pd.DataFrame): The input DataFrame with potential missing values.
    - method (str): The method for filling missing values, default is 'ffill' (forward fill).

    Returns:
    - pd.DataFrame: The DataFrame with missing values handled.
    """
    return data.fillna(method=method)