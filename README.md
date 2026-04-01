#📊 BI + ML Platform

🚀 Business Intelligence & Machine Learning Platform
An interactive data analytics web application built with Python + Streamlit that transforms raw datasets into visual dashboards, machine learning predictions, forecasts, and automated insights.

Built using Python · Streamlit · scikit-learn · Plotly · statsmodels

🌐 Live App:
https://bimlplatform-8aqd7my7y6xhlz8egwv6pl.streamlit.app/

✨ Project Overview

The BI + ML Platform is a browser-based analytics application that works like a lightweight Power BI / Tableau alternative but with an integrated Machine Learning engine.

Users can upload any dataset and automatically perform:

✔ Data Cleaning
✔ Exploratory Data Analysis (EDA)
✔ Business Intelligence Dashboards
✔ Machine Learning Model Training
✔ Predictions on new data
✔ Time-Series Forecasting
✔ Automated Business Insights

All of this happens without writing additional code once the platform is running.

🎯 Problem the Platform Solves

In many organisations, data is stored in spreadsheets but extracting insights requires either:

💰 Expensive BI tools (Power BI, Tableau)
👨‍💻 Skilled data scientists

This platform removes both barriers by providing:

✔ Open-source analytics tools
✔ Automated ML pipelines
✔ End-to-end data workflow in one application

🧠 Key Features

📂 Upload Dataset
Upload CSV or Excel datasets directly into the application.

🧹 Automatic Data Cleaning

Missing value handling
Duplicate removal
Outlier detection
Categorical encoding

📊 Exploratory Data Analysis

Distribution plots
Scatter plots
Correlation heatmaps
Boxplots by category

📈 Business Intelligence Dashboard

KPI cards
Time-series charts
Category comparison charts
Segment distribution visuals

🤖 Machine Learning Models

Random Forest
Gradient Boosting
Logistic Regression
Decision Tree

🔮 Predictions
Predict outcomes for new data with probability scores.

📉 Forecasting
Generate future predictions using ARIMA time-series forecasting.

📑 Automated Insights
Generate business insights like revenue concentration and performance gaps.

🏗 System Architecture

The platform follows a two-layer architecture:

🖥 Presentation Layer

Streamlit dashboard pages

⚙ Logic Layer

Central data manager handling:

Data loading
Cleaning
Machine learning
Forecasting
Insight generation

📁 Project Structure
BI_ML_PLATFORM
│
├── main.py
│
├── dashboard
│   ├── home.py
│   ├── upload.py
│   ├── cleaning.py
│   ├── eda.py
│   ├── bi.py
│   ├── ml.py
│   ├── predictions.py
│   ├── forecasting.py
│   └── insights.py
│
├── utils
│   └── data_manager.py
│
└── requirements.txt

data_manager.py acts as the core engine of the platform containing all logic.

⚙ Technology Stack
Technology	Purpose
🐍 Python	Core programming language
🎈 Streamlit	Web application framework
🐼 Pandas	Data manipulation
🔢 NumPy	Numerical computation
🤖 scikit-learn	Machine learning models
📊 Plotly	Interactive visualizations
📉 statsmodels	ARIMA forecasting

All libraries used are free and open-source.

📊 Machine Learning Models

The platform trains multiple models automatically and compares performance.

Model	Type
Random Forest	Classification
Gradient Boosting	Classification
Logistic Regression	Classification
Decision Tree	Classification
Linear Regression	Regression
K-Means	Clustering

Evaluation metrics used:

✔ Accuracy
✔ F1 Score
✔ AUC-ROC

🔮 Forecasting

Time-series forecasting is implemented using ARIMA models.

The forecasting pipeline:

1️⃣ Aggregate data by time period
2️⃣ Train ARIMA model
3️⃣ Generate predictions
4️⃣ Display confidence intervals

If ARIMA fails, the system uses a linear regression fallback.

📄 Insight Generation

The platform automatically generates insights such as:

📉 Revenue concentration (Pareto analysis)
📊 Segment performance gaps
⚠ Risk alerts for high churn rate
📈 Trend analysis across time

These insights help translate data into actionable business decisions.

▶️ How to Run the Project
1️⃣ Clone the repository
git clone https://github.com/khushbirkaur/BI_ML_PLATFORM.git
2️⃣ Navigate to the folder
cd BI_ML_PLATFORM
3️⃣ Install dependencies
pip install -r requirements.txt
4️⃣ Run the application
streamlit run main.py

Then open:

http://localhost:8501
🎥 Recommended Demo Flow

1️⃣ Upload sample dataset
2️⃣ Run data cleaning pipeline
3️⃣ Explore EDA visualisations
4️⃣ View BI dashboard metrics
5️⃣ Train ML models
6️⃣ Generate predictions
7️⃣ Run forecasting
8️⃣ Generate insight report

This flow demonstrates the complete analytics pipeline.

👩‍💻 Author

Khushbir Kaur Bamrah
Data Science Student

📧 khushbir36@gmail.com

💼 LinkedIn: linkedin.com/in/khushbir-kaur-bamrah-178aa4269
💻 GitHub: github.com/khushbirkaur
