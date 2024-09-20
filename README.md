# AlphaCare-Insurance-Analytics

## Project Overview
This repository contains the analysis of historical insurance claim data for AlphaCare Insurance Solutions (ACIS). The project aims to optimize the marketing strategy and identify "low-risk" targets for potential premium reductions, ultimately attracting new clients.

## Business Objectives
1. Analyze historical insurance claim data.
2. Optimize the marketing strategy.
3. Discover "low-risk" targets for premium reduction.
4. Attract new clients.

## Key Components
1. **Exploratory Data Analysis (EDA)**
2. **A/B Hypothesis Testing**
3. **Statistical Modeling**

## Project Structure
```bash
Telco-Telecom-Analysis/
│
├── .vscode/                 # VSCode settings
│   └── settings.json
│
├── .github/                 # GitHub Actions
│   └── workflows/
│       └── unittests.yml    # GitHub Actions CI for unittests
│
├── .gitignore               # Files and folders to be ignored by git
├── requirements.txt         # Contains dependencies for the project
├── README.md                # Documentation for the project
│
├── src/                     # Source files
│   ├── __init__.py
│   └── notebooks/
│       ├── __init__.py
│       ├── task_1.ipynb     # Jupyter notebook for data cleaning and EDA analysis
│       ├── task_3.ipynb  # Jupyter notebook for hypothesis testing analysis
│       └── task_4.ipynb  # Jupyter notebook for statistical modeling analysis
│
├── tests/                   # Unit test files
│   ├── __init__.py
│   └── test_data_processing.py
│
└── scripts/                 # Scripts for data processing and analysis
    ├── data_processing.py       # Contains script for data cleaning and EDA analysis
    ├── hypothesis_analysis.py   # Contains a script file for A/B hypothesis testing
    └── statistical_modeling.py  # Contains a script file for statistical modeling analysis
```

## Setup and Installation

### 1. Clone this repository:
```bash
git clone https://github.com/nebiyu-ethio/AlphaCare-Insurance-Analytics
cd AlphaCare-Insurance-Analytics
```

### 2. Create a virtual environment:
```bash
python -m venv .env
source .env/bin/activate  # On Windows, use `.env\Scripts\activate`
```

### 3. Install required packages:
```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Activate the virtual environment:
```bash
source .env/bin/activate  # On Windows, use `.env\Scripts\activate`
```

### 2. Launch Jupyter Notebook:
```bash
jupyter notebook
```

### 3. Open the notebooks in the `notebooks/` directory to view or run the analyses.