import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import zipfile
sys.path.append(os.path.abspath(os.path.join('..')))
from scripts.hypothesis_analysis import *

def app():
    # Load Data
    zip_file_path = 'cleaned_insurance_data.zip'
    csv_file_name = 'cleaned_insurance_data.csv' 
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        with z.open(csv_file_name) as f:
            data = pd.read_csv(f, low_memory=False, index_col=False)

    # Calculate risk and margin
    data['Margin'] = data['TotalPremium'] - data['TotalClaims']

    def print_test_results(result, risks):
        if 'error' in result:
            print(result['error'])
        else:
            print(f"Test type: {result['test_type']}")
            print(f"Statistic: {result['statistic']}")
            print(f"p-value: {result['p_value']}")
            print(result['interpretation'])
        print(f"Risks:\n{risks}\n")

    # **1. Test for risk differences across provinces**
    # - **Null Hypothesis (H₀)**: There are no risk differences across provinces (interms of TotalPremium)
    # - **Alternative Hypothesis (H₁)**: There is risk differences across provinces

    # Test for Risk Differences across Provinces using Anova Test

    print("1. Testing for risk differences across provinces")
    province_risks = calculate_risk(data, 'Province', 'TotalPremium')
    result = perform_statistical_test(data, 'Province', 'TotalPremium', 'anova')
    print_test_results(result, province_risks)

    # Test for risk differences across provinces using chi_square test

    print("1. Testing for risk differences across provinces")
    province_risks = calculate_risk(data, 'Province', 'TotalPremium')
    result = perform_statistical_test(data, 'Province', 'TotalPremium', 'chi_square')
    print_test_results(result, province_risks)

    # Visualizations for Risk Difference accross province

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Province', y='TotalPremium', data=data)
    plt.title('Distribution of Risk by Province')
    plt.xticks(rotation=45)
    plt.show()
    plt.close()
    
    # **2. Test for risk differences between zip codes**
    # - **Null Hypothesis (H₀)**: TThere are no risk differences between zip codes(interms of TotalPremium)
    # - **Alternative Hypothesis (H₁)**: There is risk differences between zip codes

    # Test for risk differences between zipcodes using anova test 
    print("2. Testing for risk differences between zipcodes")
    zipcode_risks = calculate_risk(data, 'PostalCode', 'TotalPremium')
    result = perform_statistical_test(data, 'PostalCode', 'TotalPremium', 'anova')
    print_test_results(result, zipcode_risks.nlargest(5))

    # Test for risk differences between zipcodes using chi_square test 
    print("2. Testing for risk differences between zipcodes")
    zipcode_risks = calculate_risk(data, 'PostalCode', 'TotalPremium')
    result = perform_statistical_test(data, 'PostalCode', 'TotalPremium', 'chi_square')
    print_test_results(result, zipcode_risks.nlargest(5))

    # **3. Test for margin (profit) differences between zip codes**
    # - **Null Hypothesis (H₀)**: There are no significant margin (profit) difference between zip codes
    # - **Alternative Hypothesis (H₁)**: There is a significant margin (profit) difference between zip codes

    # Test for margin (profit) differences between zip codes using anova test 
    print("3. Testing for margin differences between zip codes")
    zipcode_margins = calculate_margin(data, 'PostalCode')
    result = perform_statistical_test(data, 'PostalCode', 'Margin', 'anova')
    print_test_results(result, zipcode_margins.nlargest(5))

    # **4. Test for Risk Differences Between Women and Men**
    # - **Null Hypothesis (H₀)**: There is no significant difference in risk between males and females in terms of Total Premium.  
    # - **Alternative Hypothesis (H₁)**: There is a significant difference in risk between males and females.

    # Test for risk differences between Women and Men
    print("4. Testing for risk differences between Women and Men")
    filtered_data = data[data['Gender'].isin(['Male', 'Female'])]
    gender_risks = calculate_risk(filtered_data, 'Gender', 'TotalPremium')
    result = perform_statistical_test(filtered_data, 'Gender', 'TotalPremium', 't_test')
    print_test_results(result, gender_risks)

    # Visualizations for Risk difference between Men and Women
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Gender', y='TotalPremium', data=filtered_data)
    plt.title(' Risk difference by Gender')
    plt.show()
    plt.close()

    # **===> Observation**
    #  - Since the p-value is much smaller than the typical alpha level (0.05), the test results correctly indicate the rejection of the null hypothesis. Even though the risk difference appears small visually, statistical significance can still be present due to factors such as sample size or data distribution. While the p-value suggests that the difference is statistically significant, the actual difference in the means seems minimal.

    # Analysis of Risk vs. Premium correlation
    data['Risk'] = data['TotalClaims'] / data['SumInsured']
    data['PremiumRate'] = data['TotalPremium'] / data['SumInsured']
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Risk', y='PremiumRate', data=data)
    plt.title('Risk vs. Premium Rate')
    plt.xlabel('Risk (Total Claims / Sum Insured)')
    plt.ylabel('Premium Rate (Total Premium / Sum Insured)')
    plt.show()
    plt.close()