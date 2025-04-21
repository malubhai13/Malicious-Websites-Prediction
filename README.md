# Malicious Website Detection

## Project Overview
This project implements a machine learning model to detect and classify malicious websites based on various features like URL characteristics, content details, and WHOIS information. Using a Random Forest classifier, the system achieves high accuracy in distinguishing between benign and malicious websites.

## Dataset
The dataset contains website data with the following key features:
- URL length and characteristics
- Content length
- DNS query times
- Special character counts
- Server information
- WHOIS data (registration date, country, etc.)
- Website type (benign or malicious) as the target variable

## Data Preprocessing Steps
- Handling missing values in features like CONTENT_LENGTH, DNS_QUERY_TIMES, and SERVER
- Converting WHOIS date columns to datetime format
- Encoding categorical variables using LabelEncoder
- Removing duplicates and irrelevant columns

## Exploratory Data Analysis
The project includes comprehensive EDA with visualizations:
- Distribution of benign vs malicious websites
- Top countries in the dataset
- Server type distributions
- Pairplots showing relationships between key features
- Website registrations trend over time
- Average content length by year
- URL length distributions
- Box plots for special characters by server type
- Correlation heatmap

## Statistical Analysis
- Outlier detection using IQR and Z-score methods
- Descriptive statistics for content length
- T-test analysis comparing content length between benign and malicious websites
- Correlation and covariance analysis

## Machine Learning Model
The project uses a **Random Forest Classifier** for the following reasons:
- Effectiveness with mixed data types
- Robustness to outliers
- Good performance on potentially imbalanced datasets
- Feature importance insights
- Reduced overfitting through ensemble learning

## Model Evaluation
The model achieves excellent performance metrics:
- Accuracy: 95.00%
- Precision: 0.91
- Recall: 0.87
- F1 Score: 0.89
- R² Score: 0.86
- Mean Squared Error (MSE): 0.05
- Root Mean Squared Error (RMSE): 0.22
- Mean Absolute Error (MAE): 0.05

## Key Visualizations
- Confusion matrix
- Actual vs predicted label counts
- Classification report

## Conclusion
The Random Forest model demonstrates strong capability in distinguishing between benign and malicious websites. The analysis reveals important patterns in URL characteristics, registration patterns, and geographical distribution of malicious websites. The model achieves a good balance between precision and recall, which is crucial for security applications where both false positives and false negatives can be problematic.

## Requirements
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy

## Usage
1. Ensure all dependencies are installed
2. Place the dataset file in the appropriate directory
3. Run the script to perform data preprocessing, visualization, model training, and evaluation

## Future Work
- Hyperparameter tuning for model optimization
- Feature engineering to improve predictive performance
- Exploration of alternative algorithms
- Addressing potential class imbalance
- Development of a real-time detection system
