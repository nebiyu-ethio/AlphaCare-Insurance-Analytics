import streamlit as st
from apps.task_1 import app as EDA_app
from apps.task_3 import app as AB_Hypothesis_Testing_app
from apps.task_4 import app as Statistical_Modeling_app

class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({"title": title, "function": func})

    def run(self):
        app = st.sidebar.selectbox("Select Analysis", self.apps, format_func=lambda app: app['title'])
        app['function']()

# Create the app
multi_app = MultiApp()

# Add all your apps here
multi_app.add_app("Exploratory Data Analysis (EDA)", EDA_app)
multi_app.add_app("A/B Hypothesis Testing", AB_Hypothesis_Testing_app)
multi_app.add_app("Statistical Modeling", Statistical_Modeling_app)

# Run the app
multi_app.run()