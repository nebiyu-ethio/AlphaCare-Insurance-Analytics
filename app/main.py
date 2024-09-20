import streamlit as st
from multiapp import MultiApp

# Import the converted notebooks as modules
from apps import task_1
from apps import task_3
from apps import task_4

# Create a multi-page app
app = MultiApp()

# Title of the dashboard
st.title("AlphaCare Insurance Analytics")

# Add all your applications here
app.add_app("Exploratory Data Analysis (EDA)", task_1.app)
app.add_app("A/B Hypothesis Testing", task_3.app)
app.add_app("Statistical Modeling", task_4.app)

# Run the app
app.run()