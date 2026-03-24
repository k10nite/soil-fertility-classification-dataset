import pandas as pd

# Load the merged and sorted DataFrame
input_path = '/content/final_merged_data_sorted.csv'
df_to_clean = pd.read_csv(input_path)

print("Original DataFrame shape:", df_to_clean.shape)
print("Original DataFrame head:")
display(df_to_clean.head())

# Define the columns to check for null values
columns_to_check = ['n', 'p', 'k', 'ph']

# Drop rows where any of the specified columns have NaN values
df_cleaned = df_to_clean.dropna(subset=columns_to_check)

print("\nCleaned DataFrame shape:", df_cleaned.shape)
print("Cleaned DataFrame head:")
display(df_cleaned.head())

print("Summary Statistics for df_cleaned (Numerical Columns):")
display(df_cleaned.describe())

print("\nDataFrame Information for df_cleaned:")
df_cleaned.info()

# Save the cleaned DataFrame to a new CSV file
output_path_cleaned = '/content/final_merged_data_cleaned.csv'
df_cleaned.to_csv(output_path_cleaned, index=False)
print(f"\nCleaned data saved to: {output_path_cleaned}")