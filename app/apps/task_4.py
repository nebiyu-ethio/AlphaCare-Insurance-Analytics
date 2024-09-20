import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import zipfile
sys.path.append(os.path.abspath(os.path.join('..')))
from scripts.statistical_modelling import *

def app():
    # Load Data
    zip_file_path = 'cleaned_insurance_data.zip'
    csv_file_name = 'cleaned_insurance_data.csv'  # Name of the file inside the zip

    with zipfile.ZipFile(zip_file_path, 'r') as z:
        with z.open(csv_file_name) as f:
            data = pd.read_csv(f, low_memory=False, index_col=False)

    models = Modelling(data)
    # Specify the numeric and categorical features to use 
    numeric_features = ['SumInsured', 'CalculatedPremiumPerTerm','RegistrationYear','PostalCode'] 
    categorical_features = ['Province', 'CoverType', 'VehicleType', 'make', 'Gender', 'MaritalStatus','PostalCode','Model','CoverCategory','NewVehicle','RegistrationYear','Citizenship' ]
    features = list(set(numeric_features) | set(categorical_features) - set(['TotalPremium', 'TotalClaims']))

    # Encode categorical variables using label encoder
    for col in categorical_features:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    X = data[features]
    y_premium = data['TotalPremium']
    y_claims= data['TotalClaims']

    # split the data
    X_train, X_test, y_claims_train, y_claims_test=models.split_data(X,y_claims)
    X_train, X_test, y_premium_train, y_premium_test=models.split_data(X,y_premium)

    # Model Building - Train Linear Regression Model
    model_claim = LinearRegression()
    model_premium=LinearRegression()
    model_claim.fit(X_train, y_claims_train) 
    model_premium.fit(X_train, y_premium_train)

    print ('Totalclaims Linear Regression result')
    models.model_testing(model_claim, X_test, y_claims_test)
    print('')
    print('Total Premium Linear Regression Result')
    models.model_testing(model_premium, X_test,y_premium_test)

    # Model Training for TotalClaims
    Linear_Regression = LinearRegression()
    Linear_Regression .fit(X_train, y_claims_train) 

    Decision_Tree = DecisionTreeRegressor(random_state=42)
    Decision_Tree.fit(X_train, y_claims_train)

    Random_Forest =RandomForestRegressor(n_estimators=100, random_state=42)
    Random_Forest.fit(X_train, y_claims_train)

    XGBoost= XGBRegressor(n_estimators=100, random_state=42)
    XGBoost.fit(X_train, y_claims_train)

    model_names = {
        Linear_Regression: "Linear Regression",
        Decision_Tree: "Decision Tree",
        Random_Forest: "Random Forest",
        XGBoost: "XGBoost"
    }

    for model in [Linear_Regression, Decision_Tree, Random_Forest, XGBoost]:
        print(f"{model_names[model]} Model result for TotalClaims")
    
    # Perform model testing
        models.model_testing(model, X_test, y_claims_test)
        print()

    # Model Training for TotalPremium
    Linear_Regression = LinearRegression()
    Linear_Regression .fit(X_train, y_premium_train) 

    Decision_Tree = DecisionTreeRegressor(random_state=42)
    Decision_Tree.fit(X_train, y_premium_train)

    Random_Forest =RandomForestRegressor(n_estimators=100, random_state=42)
    Random_Forest.fit(X_train, y_premium_train)

    XGBoost= XGBRegressor(n_estimators=100, random_state=42)
    XGBoost.fit(X_train, y_premium_train)

    model_names = {
        Linear_Regression: "Linear Regression",
        Decision_Tree: "Decision Tree",
        Random_Forest: "Random Forest",
        XGBoost: "XGBoost"
    }

    for model in [Linear_Regression, Decision_Tree, Random_Forest, XGBoost]:
        print(f"{model_names[model]} Model result for Totalpremium")
        print('----------------------------------------------------')
        # Perform model testing
        models.model_testing(model, X_test, y_premium_test)
        print()

    # **===> Analysis**
    # 
    # 1. **Accuracy (R²):**
    #    - Random Forest performs best with an R² of 0.6681, closely followed by Decision Tree (0.6671).
    #    - XGBoost is third with 0.5275, while Linear Regression performs the worst with 0.3551.
    # 
    # 2. **Error Metrics (MAE, MSE, RMSE):**
    #    - Decision Tree and Random Forest have the lowest error rates across all metrics.
    #    - XGBoost performs better than Linear Regression but worse than the other tree-based models.
    #    - Linear Regression has the highest error rates, further indicating its unsuitability for this problem.
    # 
    # 3. **Model Comparison:**
    #    - The Random Forest model appears to be the best choice for predicting Total Premium, with the Decision Tree model as a close second.

    # Scatter Plot for Each Model: Predicted vs. Actual Values
    for model in [Linear_Regression, Decision_Tree, Random_Forest, XGBoost]:
        print(f"Plotting for {model_names[model]}")
        plot_predictions_vs_actuals(model, X_test, y_premium_test)
        print()

    # Plot all models' predictions on a single scatter plot to visually compare their performance.
    models_list = [
        (Linear_Regression, "Linear Regression"),
        (Decision_Tree, "Decision Tree"),
        (Random_Forest, "Random Forest"),
        (XGBoost, "XGBoost")
    ]
    plot_all_models_predictions(models_list, X_test, y_premium_test, model_names)

    # Initialize SHAP explainer

    explainer = shap.Explainer(XGBoost)
    shap_values = explainer(X_test)
    # Waterfall plot for the first observation
    shap.waterfall_plot(shap_values[0])

    # Plot SHAP values
    shap.summary_plot(shap_values, X_test)
    shap.plots.bar(shap_values)