import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import zipfile
sys.path.append(os.path.abspath(os.path.join('..')))
from scripts.data_processing import *

def app():
    # Load Data 
    zip_file_path = os.path.join(os.path.dirname(__file__), 'MachineLearningRating_v3.zip')
    csv_file_name = 'MachineLearningRating_v3.txt'  # Name of the file inside the zip

    with zipfile.ZipFile(zip_file_path, 'r') as z:
        with z.open(csv_file_name) as f:
            data = pd.read_csv(f, delimiter='|', low_memory=False)
    
    data.head()

    # Properties of the Data
    print(data.info())

    # **There are 37 categorical columns and 15 numerical columns.** <br>
    #     ===> bool-type -1 <br>
    #     ===> float-type - 11 <br>
    #     ===> int-type - 4 <br>
    #     ===> object - 36 <br>

    data.describe()

    data.columns

    data.dtypes

    # Checking the Quality of Data

    # Checking Null Values
    print(data.isnull().sum())

    # Number of Duplicate Values
    print(f"Number of Duplicate rows: {data.duplicated().sum()}")

    # Missing Values
    calculate_missing_percentage(data)

    missing_data = check_missing_values(data)
    missing_data
    
    # The dataframe contains 52 columns, of which 22 have missing values. <br>
    # 
    # **Handling Missing values**<br>
    # Drop columns with a high percentage of missing values and impute remaining missing values.<br>
    # 
    # Drop columns with high percentage of missing values (>50%): <br>
    # ===> NumberOfVehiclesInFleet(100% missing)<br>
    # ===> CrossBorder (99.93 %)<br>
    # ===> CustomValueEstimate(77.96%)<br>
    # ===> Rebuilt(64.18%)<br>
    # ===> Converted(64.18%)<br>
    # ===> WrittenOff(64.18%)<br>

    data_cleaned= drop_high_missing_columns(data)

    # **For the remaining data:**<br>
    # ===>For categorical columns, impute missing values using the mode (most frequent value).<br>
    # ===>For numerical columns, use the median to fill in missing values.<br>
    
    data= impute_missing_values(data_cleaned)

    # **Checking the missing values of the cleaned data.**
    missing_data = check_missing_values(data)
    missing_data
    
    # **Convert the 'VehicleIntroDate' from an object to datetime format to perform time-based analysis.**
    data['VehicleIntroDate'] = pd.to_datetime(data['VehicleIntroDate'], format='mixed', utc=True)
    
    # Missing Data Imputation
    data['Title'].unique()
    
    data['Gender'].unique()

    count = (data['Gender'] == 'Not specified').sum()
    print(count)

    # Define mappings for Title to Gender
    title_to_gender = {
        'Mr': 'Male',
        'Mrs': 'Female',
        'Miss': 'Female',
        'Ms': 'Female',
    }

    # Update Gender based on Title where Gender is 'not specified'
    data.loc[(data['Gender'] == 'Not specified') & (data['Title'] != 'Dr'), 'Gender'] = data['Title'].map(title_to_gender)
    
    count = (data['Gender'] == 'Not specified').sum()
    print(count)
    
    # **Univariate Analysis**
    
    # Plot histograms for numerical columns and bar charts for categorical columns to understand distributions for the selected key columns from the dataset. <br>
    # 
    # **Distribution of numerical columns**
    numerical_columns= ['TotalPremium',	 'TotalClaims', 'SumInsured', 'CalculatedPremiumPerTerm']

    # Plot histograms for key numerical columns
    print("Histograms for Numerical Columns:")
    for col in numerical_columns:
        plot_histogram(data, col)
        plt.close()

    # **Distribution of categorical columns**
    categorical_columns=['Province', 'VehicleType',  'Gender', 'MaritalStatus', 'NewVehicle', 'RegistrationYear','Citizenship' ]
    
    # Plot bar charts for categorical columns
    for col in categorical_columns:
        plot_bar_chart(data, col)
        plt.close()

    # **Insights from the univariate analysis of numeric columns:**<br>
    # 
    # - There is a wide distribution across different postal codes, indicating geographical diversity in the customer base.<br>
    # - Certain vehicle types (such as passenger vehicles) and makes (e.g., Toyota) are overrepresented in the data, suggesting these vehicles are insured more frequently.<br>
    # - Males tend to be more frequently represented as policyholders than females.<br>

    # # Bivariate Analysis
    # 
    # Use scatter plots and correlation heatmaps to understand relationships between variables.
    plot_scatter(data, 'TotalPremium', 'TotalClaims')
    plt.close()

    plot_scatter(data, 'NewVehicle', 'Province')
    plt.close()

    # **Examine the correlation between different attributes in our dataset.**
    plot_correlation_heatmap(data, ['TotalPremium', 'TotalClaims'])
    plt.close()

    # ===> **Based on the plot, it can be observed that the correlation between TotalPremium and TotalClaims is positive but weak, with a correlation coefficient of 0.12. This suggests that higher premiums generally result in higher claims, though there is significant variability.**

    # Correlation between numerical columns
    plot_correlation_heatmap(data, numerical_columns)
    plt.close()

    # **===> From the correlation heatmap, we can see that:**
    # 
    # - TotalPremium is strongly correlated with CalculatedPremiumPerTerm (0.64).
    # - TotalClaims shows weak correlations with the other variables, with a slight positive correlation to TotalPremium (0.12).
    # - SumInsured shows almost no significant correlation with the other variables.

    # Multivariate Analysis
    plot_scatter(data, 'TotalPremium', 'TotalClaims', 'Province')
    plt.close()

    # Correlation matrix
    correlation_matrix_province = data.groupby('Province')[['TotalPremium', 'TotalClaims']].corr().unstack()['TotalClaims']['TotalPremium']
    print("Correlation Matrix of TotalPremium vs TotalClaims by Province")
    print(correlation_matrix_province)

    # **===> From the Multivariate Analysis:**
    # 
    # - There are noticeable differences in claim patterns across provinces. For example, Gauteng and Western Cape exhibit the highest claims relative to premiums.
    # - The majority of data points (across most provinces) are clustered near the lower range of the TotalPremium axis, indicating generally lower premiums.
    # - Provinces like Western Cape, which have both high TotalPremium and high TotalClaims, imply higher risks in those areas.

    columns=['TotalPremium', 'TotalClaims', 'MaritalStatus', 'Gender']
    columns_data = data[columns]

    columns_data = pd.get_dummies(columns_data, columns=['MaritalStatus', 'Gender'])
    plot_correlation_heatmap(columns_data, columns_data.columns)
    plt.close()
    # The correlation between "Gender_Female" and "TotalPremium" is approximately -0.00074. This indicates a very weak negative correlation, suggesting that being female has almost no direct linear impact on the total premium paid.<br>
    # **Correlation with Gender_Male:** Similarly, the correlation between "Gender_Male" and "TotalPremium" is approximately 0.00087, indicating an almost negligible positive correlation. This means being male also shows no significant linear effect on the total premium.<br>
    # **===> Overall Analysis:** There seems to be almost no difference between male and female customers regarding the total premium paid.<br>
    # 
    # The correlation between "Gender_Female" and "TotalClaims" is approximately -0.0021, suggesting a very weak negative relationship. This implies that female customers might slightly tend to have fewer claims, but the effect is negligible.<br>
    # The correlation between "Gender_Male" and "TotalClaims" is approximately 0.0021, indicating a very slight positive relationship. This suggests that male customers might have marginally more claims.<br>
    # **===> Overall Analysis:** The correlations indicate that there is no substantial difference in the number of claims between male and female customers. The correlations are extremely close to zero, implying no strong linear relationship.<br>

    # Data Comparison
    trend_over_geography(data, 'Province', 'TotalPremium')
    trend_over_geography(data, 'Province', 'TotalClaims')

    trend_over_geography(data, 'CoverType', 'TotalPremium')

    # Outlier Detection
    for col in ['TotalPremium', 'TotalClaims']:
        plot_boxplot(data, 'Province', col)
        plt.close()

    data_numerical_cols = data[numerical_columns]
    outlier_box_plots(data_numerical_cols)

    # **===> Insights from the outlier detection:**
    # 
    # - There are significant outliers in TotalPremium and TotalClaims, with some extremely high values.
    # - SumInsured and CalculatedPremiumPerTerm also show outliers, though they are less extreme compared to TotalPremium and TotalClaims.