import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 


# calculate percentage of missing values
def calculate_missing_percentage(dataframe):
    # Determine the total number of elements in the DataFrame
    total_elements = np.prod(dataframe.shape)

    # Count the number of missing values in each column
    missing_values = dataframe.isna().sum()

    # Sum the total number of missing values
    total_missing = missing_values.sum()

    # Compute the percentage of missing values
    percentage_missing = (total_missing / total_elements) * 100

    # Print the result, rounded to two decimal places
    print(f"The dataset has {round(percentage_missing, 2)}% missing values.")


def check_missing_values(data):
    """Check for missing values in the dataset."""
    missing_values = data.isnull().sum()
    missing_percentages = 100 * data.isnull().sum() / len(data)
    column_data_types = data.dtypes
    missing_table = pd.concat([missing_values, missing_percentages, column_data_types], axis=1, keys=['Missing Values', '% of Total Values','Data type'])
    return missing_table.sort_values('% of Total Values', ascending=False).round(2)


def drop_high_missing_columns(data, threshold=50):
    missing_series = data.isnull().sum() / len(data) * 100
    columns_to_drop = missing_series[missing_series > threshold].index
    data_cleaned = data.drop(columns=columns_to_drop)
    print(f"Dropped columns: {list(columns_to_drop)}")
    return data_cleaned

def impute_missing_values(data):
    for column in data.columns:
        if data[column].dtype == 'object':
            # Categorical column: impute with mode
            mode_value = data[column].mode()[0]
            #data[column].fillna(mode_value)
            data[column] = data[column].fillna(mode_value)

        else:
            # Numerical column: impute with median
            median_value = data[column].median()
            #data[column].fillna(median_value)
            data[column] = data[column].fillna(median_value)

    return data


# Function to plot histogram for numerical columns
def plot_histogram(data, column):
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()


# Function to plot bar chart for categorical columns
def plot_bar_chart(data, column):
    plt.figure(figsize=(12, 6))
    data[column].value_counts().plot(kind='bar')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

# function to perform correlation heatmap for key numerical columns
def plot_correlation_heatmap(data, columns):
    corr = data[columns].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()


def plot_scatter(data, x, y, hue=None):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=data, x=x, y=y, hue=hue)
    plt.title(f'{y} vs {x}')
    plt.show()


def plot_boxplot(data, x, y):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data, x=x, y=y)
    plt.title(f'Boxplot of {y} by {x}')
    plt.xticks(rotation=45)
    plt.show()


def cap_outliers(data, columns=None):
    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    data_capped = data.copy()
    
    if columns is None:
        columns = data_capped.select_dtypes(include=[np.number]).columns
    
    for column in columns:
        Q1 = data_capped[column].quantile(0.25)
        Q3 = data_capped[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        data_capped.loc[data_capped[column] < lower_bound, column] = lower_bound
        data_capped.loc[data_capped[column] > upper_bound, column] = upper_bound
    
    return data_capped


# outlier plot for each individual numeric column
def outlier_box_plots(data):
    for column in data:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=data[column])
        plt.title(f'Box plot of {column}')
        plt.show()


def trend_over_geography(data, geography_column, value_column):
    grouped = data.groupby(geography_column)[value_column].mean().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    grouped.plot(kind='bar')
    plt.title(f'Average {value_column} by {geography_column}')
    plt.xticks(rotation=45)
    plt.show()