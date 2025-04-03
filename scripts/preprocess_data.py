import argparse
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process an Excel file.')
    parser.add_argument('--file_path', type=str, default='BusinessDS.xlsx', help='Path to the Excel file')
    args = parser.parse_args()

    # Read the data from the Excel file
    try:
        df = pd.read_excel(args.file_path, parse_dates=['Date'])
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # === Task 1: Report general statistics ===
    print("\n=== General Statistics ===")
    num_samples = df.shape[0]  # Number of rows (samples)
    num_features = df.shape[1]  # Number of columns (features)
    print(f"Number of samples (rows): {num_samples}")
    print(f"Number of features (columns): {num_features}")
    print("\nDataset Info:")
    print(df.info())
    print("\nDescriptive Statistics:")
    print(df.describe(include='all'))

    # === Task 2: Report categorical and continuous features with appropriate charts ===
    print("\n=== Categorical and Continuous Features ===")
    categorical_features = df.select_dtypes(include=['object', 'category']).columns
    continuous_features = df.select_dtypes(include=['int64', 'float64']).columns

    print("Categorical Features:", categorical_features)
    print("Continuous Features:", continuous_features)

    # Visualize categorical features
    for feature in categorical_features:
        plt.figure(figsize=(10, 4))
        sns.countplot(data=df, x=feature)
        plt.title(f'Distribution of {feature}')
        plt.show()

    # Visualize continuous features
    for feature in continuous_features:
        plt.figure(figsize=(10, 4))
        sns.histplot(df[feature], kde=True)
        plt.title(f'Distribution of {feature}')
        plt.show()

    # === Task 3: Find outliers with two methods (Z-Score and IQR) ===
    print("\n=== Outlier Detection ===")
    # Z-Score Method
    z_scores = df[continuous_features].apply(zscore)  # Calculate Z-Score for each continuous feature
    outliers_z = df[(z_scores.abs() > 3).any(axis=1)]  # Identify rows with any Z-Score > 3
    print("Outliers detected using Z-Score:")
    print(outliers_z)

    # IQR Method
    Q1 = df[continuous_features].quantile(0.25)
    Q3 = df[continuous_features].quantile(0.75)
    IQR = Q3 - Q1
    outliers_iqr = df[((df[continuous_features] < (Q1 - 1.5 * IQR)) | (df[continuous_features] > (Q3 + 1.5 * IQR)))].any(axis=1);
    print("\nOutliers detected using IQR:")
    print(outliers_iqr)

    # === Task 4: Report statistics of missing values and treat them ===
    print("\n=== Missing Values ===")
    missing_values = df.isnull().sum()
    print("Missing Values:")
    print(missing_values)

    # Treat missing values (fill with median for continuous features)
    df_filled = df.copy()
    for feature in continuous_features:
        df_filled[feature].fillna(df_filled[feature].median(), inplace=True)
    print("\nDataset after filling missing values:")
    print(df_filled.isnull().sum())

    # === Task 5: Report probable data cleaning actions ===
    print("\n=== Probable Data Cleaning Actions ===")
    print("1. Handle missing values by filling them with median/mean.")
    print("2. Remove or cap outliers detected using Z-Score and IQR.")
    print("3. Convert categorical features to numerical using one-hot encoding or label encoding.")
    print("4. Normalize/standardize continuous features.")

    # === Task 6: Normalize the dataset ===
    print("\n=== Normalization ===")
    scaler = MinMaxScaler()
    df_normalized = df_filled.copy()
    df_normalized[continuous_features] = scaler.fit_transform(df_filled[continuous_features])
    print("\nNormalized Dataset:")
    print(df_normalized.head())

    # Save the normalized dataset to a new file
    df_normalized.to_excel('normalized_data.xlsx', index=False)
    print("\nNormalized dataset saved to 'normalized_data.xlsx'.")

    # === Task 7: Visualize important relationships between features ===
    print("\n=== Feature Relationships ===")
    # Pairplot for continuous features
    sns.pairplot(df_normalized[continuous_features])
    plt.suptitle('Pairplot of Continuous Features', y=1.02)
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_normalized[continuous_features].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

    # Scatter plot for specific relationships (e.g., Current Price vs. Reference Price)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_normalized, x='Current Price', y='Reference Price', hue='Position Type')
    plt.title('Current Price vs. Reference Price')
    plt.show()
    
    #testing library
    import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

print("All libraries installed successfully!")
import os
file_path = os.path.join(os.path.dirname(__file__), '..', 'processed_data.csv')
df = pd.read_csv(file_path)  # This always works!

if __name__ == "__main__":
    main()