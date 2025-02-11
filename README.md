# Forecast of HPG stock code of Hoa Phat Group Joint Stock Company VN-Index

Input: Write an API using the JSON library in Python to craw the stock data for HPG from the Cafef.vn website.

Output: Use the TensorFlow library in Python to build a Deep Neural Network model to predict stock prices for the next 120 days.

Summary: 
+ Method 1: Use Scheduler in Fabric Microsoft to run Pyspark Python files to automate data scraping and loading into Lakehouse, then use the DataFlow auto-update feature to transmit data to Data Warehouse, using Power Bi connection to visualize data
 + Method 2: Use Scheduler in Fabric Microsoft runs Pyspark Python file to automate scraping data into Hpg_Occurated file and forecasting data into Hpg_pre file. Then update data to Google Big Query and use Power Bi to visualize data.

Data Visualization: Use Power BI to display the stock prices over time and predict the stock price for the next 120 days.

