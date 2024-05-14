import numpy as np
import pandas as pd
import data_processing
import db_net_model

# load all data
dataset = pd.read_csv('raw data/household_power_consumption.txt', sep=';', header=0, low_memory=False, infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime'])

# choose data frequency
frequency = 'D'

# choose a number of time steps
n_steps_in, n_steps_out = 2, 1

# epochs
epochs = 300

# define the filename for the CSV file
file_name = 'dummy_actual_predicted_values.csv'

# clean the daa
normalized_data = data_processing.clean_data(dataset, frequency)

# convert into input/output
X, y = data_processing.split_sequences(normalized_data, n_steps_in, n_steps_out)

# model training and test
db_net_model.db_net_model(X, y, n_steps_in, n_steps_out, epochs, file_name)





