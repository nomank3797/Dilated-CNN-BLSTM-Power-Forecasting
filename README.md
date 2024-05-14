# Power Consumption Forecasting with DB-Net:
Welcome to the GitHub repository for power consumption forecasting using DB-Net! DB-Net is a novel dilated CNN-based multi-step forecasting model designed specifically for power consumption prediction in integrated local energy systems.

**Paper Title:** DB-Net: A novel dilated CNN based multi-step forecasting model for power consumption in integrated local energy systems

**Paper Link:** https://www.sciencedirect.com/science/article/pii/S0142061521002635

**Data Link:** https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

# Proposed Framework:

![image](https://github.com/nomank3797/Dilated-CNN-BLSTM-Power-Forecasting/assets/114480394/0d93effb-1fbe-48fc-951a-7c55fd2980c5)

The proposed framework consists of four key steps. Step 1 involves data collection, where data is recorded using smart meters. Step 2 concentrates on refining the collected data by removing anomalies and ensuring data quality. In Step 3, the refined data is encoded using a 1D Dilated Convolutional Neural Network (DCNN). Lastly, in Step 4, the encoded data is input into a Bidirectional Long Short-Term Memory (BLSTM) network. Here, it undergoes decoding, allowing the network to learn sequential patterns and generate final predictions of power consumption.

# Usage:

To run the code and perform power consumption forecasting:

1. Download the dataset from the data link.
2. Update the dataset path in the run_forecasting.py file to point to the location where you have downloaded the dataset.
3. Run the run_forecasting.py script to execute the forecasting model.

# Citation:
Khan, Noman, et al. "DB-Net: A novel dilated CNN based multi-step forecasting model for power consumption in integrated local energy systems." International Journal of Electrical Power & Energy Systems 133 (2021): 107023.
# Contact:
If you have any questions or inquiries, please don't hesitate to contact me via email at nomank3797@gmail.com.
