import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Load the dataset
data_file = '../data/airlinedelaycauses_DelayedFlights.csv'
df = pd.read_csv(data_file)
df.drop(['Unnamed: 0', 'Year', 'DayOfWeek'], 1, inplace=True)

# Stratified random sampling based on months
num_months = df.Month.unique() # List unique months
df_stratified = pd.DataFrame(columns=df.columns) # Create empty dataframe

# Loop through the strata
for month in num_months:
    df_filtered = df[df['Month'] == month] # Filter the month based on the selected strata
    stratified_rows = df_filtered.sample(frac=0.005, random_state=1) # Random sample within the strata
    df_stratified = pd.concat([df_stratified, stratified_rows]) # Add the stratified data to the new dataframe

# Resetting the index
df_stratified = df_stratified.reset_index(drop=True)

# Split numerical and categorical columns
numerical_columns = df_stratified.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = df_stratified.select_dtypes(include=['object', 'bool']).columns

# Add the pipelines together
cleaning_pipeline = ColumnTransformer(
    transformers=[
        ('numerical', SimpleImputer(strategy='constant', fill_value=0), numerical_columns),
        ('categorical', SimpleImputer(strategy='constant', fill_value='Unknown'), categorical_columns),
    ], remainder = 'passthrough'
)

# Create a new dataframe and export it to csv
cleaned_df = cleaning_pipeline.fit_transform(df_stratified)
cleaned_df = pd.DataFrame(data=cleaned_df, columns=numerical_columns.append(categorical_columns))
cleaned_df = cleaned_df[df_stratified.columns]
cleaned_df.to_csv('../data/airline_delay_dataframe.csv', index=False)

# Convert textual to numerical
for column in categorical_columns:
    translate = cleaned_df[column].unique()
    cleaned_df[column] = cleaned_df[column].apply(lambda x: np.where(translate == x)[0][0])

# Export it to csv
cleaned_df.to_csv('../data/encoded_airline_delay_dataframe.csv', index=False)

# Create classification dataframe
classification_df = cleaned_df.drop(['DayofMonth', 'DepTime', 'CRSDepTime', 'ArrTime', 'CRSArrTime', 'UniqueCarrier', 'FlightNum', 'TailNum', 'CRSElapsedTime', 'Origin', 'Dest', 'CancellationCode', 'Cancelled'], axis=1)

# Category conditions
conditions = [
    (classification_df['ArrDelay'] < 5),
    (classification_df['ArrDelay'] > 5)]

# Category names
choices = [0, 1]

# Apply new categories
classification_df['ArrDelay'] = np.select(conditions, choices)

classification_df.to_csv('../data/classification_airline_delay_dataframe.csv', index=False)