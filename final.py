import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

df = pd.read_csv("C:/Users/91800/Documents/SEM4/INT 375/Project/dataset.csv")

"""**Data Preprocessing and Cleaning**
"""

print("\nInitial Dataset Preview:")
df.head()
df.shape
df.columns
df.info()
df.describe()
df.duplicated().sum()
df.drop_duplicates(inplace=True)
fill_values = {
    'CONTENT_LENGTH': df['CONTENT_LENGTH'].median(),
    'DNS_QUERY_TIMES': 0,
    'SERVER': 'unknown'
}
df.fillna(value=fill_values, inplace=True)
df

# Convert WHOIS date columns to datetime
df['WHOIS_REGDATE'] = pd.to_datetime(df['WHOIS_REGDATE'], errors='coerce')
df['WHOIS_UPDATED_DATE'] = pd.to_datetime(df['WHOIS_UPDATED_DATE'], errors='coerce')


df['WHOIS_REGDATE'] = df['WHOIS_REGDATE'].fillna(pd.Timestamp("1970-01-01"))
df['WHOIS_UPDATED_DATE'] = df['WHOIS_UPDATED_DATE'].fillna(pd.Timestamp("1970-01-01"))

# Confirm missing values are handled
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

from sklearn.preprocessing import LabelEncoder
label_cols = ['CHARSET', 'SERVER', 'WHOIS_COUNTRY', 'WHOIS_STATEPRO']

le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col].astype(str))  # Convert to string to avoid NaN issues

if 'URL' in df.columns:
    df.drop(columns=['URL'], inplace=True)

sns.countplot(
    x='Type',
    data=df,
    hue='Type',
    legend=False,
    palette={0: 'skyblue', 1: 'salmon'}
)
plt.title('Distribution of Website Types (0 = Benign, 1 = Malicious)', fontsize=14)
plt.xlabel('Website Type')
plt.ylabel('Count')
plt.show()

top_countries = df['WHOIS_COUNTRY'].value_counts().head(10).reset_index()
top_countries.columns = ['Country', 'Count']

# Set the Country column as a categorical type for proper handling
top_countries['Country'] = top_countries['Country'].astype(str)

colors = sns.color_palette('Spectral', n_colors=10)

plt.figure(figsize=(12, 6))
bars = sns.barplot(
    data=top_countries,
    x='Country',
    y='Count',
    hue='Country',
    legend=False,
    palette=colors
)

plt.title('Top 10 WHOIS Countries in Dataset', fontsize=16, fontweight='bold')
plt.xlabel('Country', fontsize=12)
plt.ylabel('Number of Websites', fontsize=12)

plt.ylim(0, 250)
plt.yticks(range(0, 251, 50))
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

top_servers = df['SERVER'].value_counts().head(10).reset_index()
top_servers.columns = ['Server', 'Count']

colors = sns.color_palette("Set2", n_colors=len(top_servers))

plt.figure(figsize=(10, 6))
sns.barplot(x='Server', y='Count', data=top_servers,hue="Server",legend=False, palette=colors)

plt.title('Top Server Types', fontsize=16, weight='bold')
plt.xlabel('Server')
plt.ylabel('Count')

plt.xticks(rotation=30)

plt.tight_layout()
plt.show()

selected_cols = ['CONTENT_LENGTH', 'DNS_QUERY_TIMES', 'NUMBER_SPECIAL_CHARACTERS', 'Type']
sns.set(style='whitegrid', context='notebook')
pair = sns.pairplot(
    df[['URL_LENGTH', 'NUMBER_SPECIAL_CHARACTERS', 'Type']],
    hue='Type',
    palette={0: '#2ecc71', 1: '#e74c3c'},
    diag_kind='kde',
    markers=['o', 's'],
    plot_kws={'alpha': 0.6, 's': 40},
)

pair.fig.suptitle('Pairplot of Key Features by Website Type', fontsize=16, y=1.02)
plt.show()

df['WHOIS_REGDATE'] = pd.to_datetime(df['WHOIS_REGDATE'], errors='coerce')

df['Reg_Year'] = df['WHOIS_REGDATE'].dt.year
df = df[df['Reg_Year'] > 1995]
yearly_data = df.groupby(['Reg_Year', 'Type']).size().unstack()


plt.figure(figsize=(12, 6)) # Plot
plt.plot(yearly_data.index, yearly_data[0], label='Benign', marker='o', color='green', linewidth=2)
plt.plot(yearly_data.index, yearly_data[1], label='Malicious', marker='o', color='red', linewidth=2)

plt.title('Website Registrations Over Time (Benign vs Malicious)', fontsize=16)

plt.xlabel('Year of Registration', fontsize=12)
plt.ylabel('Number of Websites', fontsize=12)
plt.legend(title='Website Type')
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

avg_content = df.groupby('Reg_Year')['CONTENT_LENGTH'].mean()

plt.figure(figsize=(12, 5))
plt.plot(avg_content.index, avg_content.values, color='blueviolet', marker='s', linestyle='-', linewidth=2)
plt.title('Average Content Length of Websites by Year', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Avg Content Length', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure()
custom_palette = {0: '#1546c2', 1: '#f50a0a'}
sns.histplot(data=df, x='URL_LENGTH', hue='Type', kde=True, bins=30, palette=custom_palette)
plt.title("Distribution of URL Lengths by Website Type")
plt.xlabel("URL Length")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

#BOX PLOT
top_servers = df['SERVER'].value_counts().head(5).index
filtered_df = df[df['SERVER'].isin(top_servers)]

plt.figure(figsize=(12, 6))
sns.boxplot(x='SERVER', y='NUMBER_SPECIAL_CHARACTERS',hue="SERVER",legend=False, data=filtered_df, palette='Set2')
plt.xticks(rotation=45)
plt.title("Special Characters Count by Server Type")
plt.xlabel("Server Type")
plt.ylabel("Number of Special Characters")
plt.tight_layout()
plt.show()

top_countries = df['WHOIS_COUNTRY'].value_counts().head(5).index
filtered_df = df[df['WHOIS_COUNTRY'].isin(top_countries)]

plt.figure(figsize=(12, 6))
sns.boxplot(x='WHOIS_COUNTRY', y='DNS_QUERY_TIMES',hue="WHOIS_COUNTRY",legend=False, data=filtered_df, palette='pastel')
plt.title("DNS Query Times by WHOIS Country")
plt.xlabel("Country")
plt.ylabel("DNS Query Time")
plt.tight_layout()
plt.show()

#HEAT MAP
plt.figure(figsize=(12, 8)) # Correlation heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

#CORRELATION AND COVARIENCE
print(df[['CONTENT_LENGTH', 'NUMBER_SPECIAL_CHARACTERS']].cov())
print(df[['CONTENT_LENGTH', 'NUMBER_SPECIAL_CHARACTERS']].corr())
print("Descriptive Statistics for 'CONTENT_LENGTH':")
print(f"Mean: {df['CONTENT_LENGTH'].mean():.2f}")
print(f"Median: {df['CONTENT_LENGTH'].median():.2f}")
print(f"Standard Deviation: {df['CONTENT_LENGTH'].std():.2f}")
print(f"Minimum Value: {df['CONTENT_LENGTH'].min()}")
print(f"Maximum Value: {df['CONTENT_LENGTH'].max()}")
print(f"Mode: {df['CONTENT_LENGTH'].mode().values.tolist()}")
Q1 = df['CONTENT_LENGTH'].quantile(0.25)
Q3 = df['CONTENT_LENGTH'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['CONTENT_LENGTH'] < Q1 - 1.5 * IQR) | (df['CONTENT_LENGTH'] > Q3 + 1.5 * IQR)]
print("\nOutlier Detection:")
print(f"Q1 (25th percentile): {Q1}")
print(f"Q3 (75th percentile): {Q3}")
print(f"IQR (Q3 - Q1): {IQR}")
print(f"Number of Outliers: {len(outliers)}")

plt.figure(figsize=(8, 6))
sns.boxplot(data=df, y='CONTENT_LENGTH')
plt.title('Box Plot of Content Length with Outliers')
plt.ylabel('Content Length')
plt.show()

from scipy.stats import zscore
content_length_z = zscore(df['CONTENT_LENGTH'].dropna())
df.loc[df['CONTENT_LENGTH'].notna(), 'Z_SCORE_CONTENT'] = content_length_z
df[['CONTENT_LENGTH', 'Z_SCORE_CONTENT']].head()

outliers = df[(df['Z_SCORE_CONTENT'].abs() > 3)]
print("Number of outliers:", len(outliers))

from scipy.stats import ttest_ind

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
content_length_outliers_iqr = df[(df['CONTENT_LENGTH'] < lower_bound) | (df['CONTENT_LENGTH'] > upper_bound)]

print("Outliers using IQR (CONTENT_LENGTH):", len(content_length_outliers_iqr))

df['Z_SCORE'] = zscore(df['CONTENT_LENGTH'])
content_length_outliers_zscore = df[(df['Z_SCORE'].abs() > 3)]
print("Outliers using Z-score (CONTENT_LENGTH):", len(content_length_outliers_zscore))


# T-statistic Analysis

group1 = df[df['Type'] == 0]['CONTENT_LENGTH'].dropna()
group2 = df[df['Type'] == 1]['CONTENT_LENGTH'].dropna()

if len(group1) > 1 and len(group2) > 1:
    t_stat, p_val = ttest_ind(group1, group2, equal_var=False) #t-test
    print(f"T-Statistic: {t_stat:.4f}, P-Value: {p_val:.4f}")
else:
    print("Not enough data to perform t-test.")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ( accuracy_score, precision_score, recall_score, f1_score,r2_score, mean_squared_error, mean_absolute_error)

model = RandomForestClassifier(random_state=42) # Fit model
from sklearn.model_selection import train_test_split

# Define features and target
features = df.drop(columns=['Type', 'URL', 'WHOIS_REGDATE', 'WHOIS_UPDATED_DATE', 'Reg_Year', 'Z_SCORE', 'Z_SCORE_CONTENT'], errors='ignore')
target = df['Type']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test) # Predict

results = pd.DataFrame({'Actual': y_test,'Predicted': y_pred})
print("Prediction Comparison:")
print(results.head(10))  #first 10 predictions

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

y_test_float = y_test.astype(float)
y_pred_float = y_pred.astype(float)

r2 = r2_score(y_test_float, y_pred_float)
mse = mean_squared_error(y_test_float, y_pred_float)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_float, y_pred_float)

#**Evaluation matrices**

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malicious'], yticklabels=['Benign', 'Malicious'])
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4)) #bar plot
actual_counts = pd.Series(y_test).value_counts().sort_index()
predicted_counts = pd.Series(y_pred).value_counts().sort_index()

df_counts = pd.DataFrame({
    'Actual': actual_counts,
    'Predicted': predicted_counts
})

df_counts.plot(kind='bar', figsize=(8, 5), color=['skyblue', 'salmon'])
plt.title('Count of Actual vs Predicted Labels')
plt.xlabel('Website Type (0 = Benign, 1 = Malicious)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.grid(True)
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malicious']))

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"R² Score: {r2:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")