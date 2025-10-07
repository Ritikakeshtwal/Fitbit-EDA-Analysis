# 🩺 Fitbit EDA: Daily Activity, Sleep, Weight, and Heart Rate Analysis

This project performs Exploratory Data Analysis (EDA) on Fitbit datasets to explore user activity, sleep patterns, weight trends, and heart rate behavior. It helps visualize health insights through daily, weekly, and categorical analysis.

The main goals of this project are to clean and merge Fitbit data from multiple CSV files, analyze daily activity, calories, and step patterns, visualize sleep duration and heart rate variation, and generate automated plots and CSV reports for better insights.

### 🧰 Tools & Libraries Used
Python, Pandas, NumPy, Matplotlib, Seaborn

### 📂 Folder Structure
Fitbit_EDA_Project/
├── data_raw/ (contains raw Fitbit CSV data)
│   ├── dailyActivity_merged.csv
│   ├── sleepDay_merged.csv
│   ├── weightLogInfo_merged.csv
│   └── heartrate_seconds_merged.csv
├── files/
│   └── fitness_result/ (auto-created after running script)
│       ├── clean_daily_sleep.csv
│       ├── agg_by_date.csv
│       ├── plots/
│           ├── steps_over_time.png
│           ├── calories_vs_steps.png
│           ├── sleep_duration_distribution.png
│           ├── weekday_activity_trends.png
│           └── heart_rate_over_time.png
├── fitbit_eda.py (main Python analysis script)
└── README.md (project documentation)

🚀 How to Run
1. Place all raw Fitbit CSV files inside the data_raw folder.
2. Open terminal and run the command:
   python fitbit_eda.py
3. After execution, check the files/fitness_result/ folder for cleaned CSVs, aggregated reports, and visualized plots in PNG format.

 📈 Sample Outputs
Steps vs Calories  
Sleep Duration Trends  
Heart Rate Analysis  
Weekly Activity Comparison  

### 🏷️ Tags
#Python #EDA #Fitbit #DataAnalytics #Visualization #LabMentix #LearningByDoing
