import numpy as np
import pandas as pd
import data_processing
import db_net_model

# Load all data
# Assuming 'household_power_consumption.txt' is in the 'raw data' directory
dataset = pd.read_csv('raw data/household_power_consumption.txt', sep=';', header=0, low_memory=False,
                      infer_datetime_format=True, parse_dates={'datetime': [0, 1]}, index_col=['datetime'])

# Choose data frequency
frequency = 'D'

# Choose a number of time steps for input and output
n_steps_in, n_steps_out = 2, 1

# Number of epochs for training
epochs = 300

# Define the filename for the CSV file to save predictions
file_name = 'dummy_actual_predicted_values.csv'

# Clean the data
normalized_data = data_processing.clean_data(dataset, frequency)

# Convert data into input/output sequences
X, y = data_processing.split_sequences(normalized_data, n_steps_in, n_steps_out)

# Model training and testing
db_net_model.db_net_model(X, y, n_steps_in, n_steps_out, epochs, file_name)
