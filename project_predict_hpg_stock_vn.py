pip install tensorflow

from pyspark.sql import SparkSession
import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def get_price_function(ma_chung_khoan):
    # Thiết lập cookies và headers
    cookies = {
    '__RC': '5',
    'favorite_stocks_state': '1',
    '__R': '3',
    '__tb': '0',
    'dtdz': '9402575f-0639-4d95-ba33-51d75f546fa6',
    '__admUTMtime': '1696906255',
    '_uidcms': '2914039761731989523',
    '_uidcms': '2914039761731989523',
    '__IP': '1731989523',
    'ASP.NET_SessionId': 'l5xtkqm432a4ufwn24rhqz2q',
    '_gid': 'GA1.2.1511612829.1699246365',
    '_ga_860L8F5EZP': 'GS1.1.1699346701.57.1.1699346714.0.0.0',
    '_ga': 'GA1.1.918773242.1694002578',
    '__uif': '__uid%3A2914039761731989523%7C__ui%3A1%252C5%7C__create%3A1661403976',
    '_ga_XLBBV02H03': 'GS1.1.1699346714.7.1.1699347294.0.0.0',
    '_ga_D40MBMET7Z': 'GS1.1.1699346714.7.1.1699347294.0.0.0',
    }  
    headers = {
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    # 'Cookie': '__RC=5; favorite_stocks_state=1; __R=3; __tb=0; dtdz=9402575f-0639-4d95-ba33-51d75f546fa6; __admUTMtime=1696906255; _uidcms=2914039761731989523; _uidcms=2914039761731989523; __IP=1731989523; ASP.NET_SessionId=l5xtkqm432a4ufwn24rhqz2q; _gid=GA1.2.1511612829.1699246365; _ga_860L8F5EZP=GS1.1.1699346701.57.1.1699346714.0.0.0; _ga=GA1.1.918773242.1694002578; __uif=__uid%3A2914039761731989523%7C__ui%3A1%252C5%7C__create%3A1661403976; _ga_XLBBV02H03=GS1.1.1699346714.7.1.1699347294.0.0.0; _ga_D40MBMET7Z=GS1.1.1699346714.7.1.1699347294.0.0.0',
    'Pragma': 'no-cache',
    'Referer': 'https://s.cafef.vn/lich-su-giao-dich-vhm-1.chn',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'sec-ch-ua': '"Google Chrome";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"macOS"',
    }  # Thêm headers của bạn

    params = {
        'Symbol': ma_chung_khoan,
        'StartDate': '',
        'EndDate': '',
        'PageIndex': '1',
        'PageSize': '10000',
    }

    response = requests.get(
        'https://s.cafef.vn/Ajax/PageNew/DataHistory/PriceHistory.ashx',
        params=params,
        cookies=cookies,
        headers=headers,
    )
    
    data = json.loads(response.text)

    if data['Success'] and data['Data'] is not None:
        result = []
        for i in data['Data']['Data']:
            d = {
                'Ngay': i['Ngay'],
                'GiaDieuChinh': i['GiaDieuChinh'],
                'GiaDongCua': i['GiaDongCua'],
                'ThayDoi': i['ThayDoi'],
                'KhoiLuongKhopLenh': i['KhoiLuongKhopLenh'],
                'GiaTriKhopLenh': i['GiaTriKhopLenh'],
                'GiaMoCua': i['GiaMoCua'],
                'GiaCaoNhat': i['GiaCaoNhat'],
                'GiaThapNhat': i['GiaThapNhat'],
            }
            result.append(d)

        df = pd.DataFrame(result)
        df['Rate'] = df['ThayDoi'].str.split('(', expand=True)[1].str.replace(' %','').str.replace(')','').astype(float)
        df['Ngay'] = pd.to_datetime(df['Ngay'], format='%d/%m/%Y')
        return df
    else:
        print("Lỗi hoặc dữ liệu không hợp lệ:", data['Message'])
        return pd.DataFrame()  # Trả về DataFrame rỗng nếu không có dữ liệu

df_HPG = get_price_function('HPG')

if not df_HPG.empty:
    df_HPG = df_HPG[['Ngay','GiaDieuChinh','GiaMoCua','GiaThapNhat','GiaCaoNhat','GiaDongCua']]
    df_HPG.rename(columns={'Ngay': 'Date', 'GiaDieuChinh': 'Price'}, inplace=True)

# Tạo Spark session với Delta
spark = SparkSession.builder \
    .appName("Predict_HPG") \
    .config("spark.sql.extensions", "delta.sql.DeltaSparkSessionExtensions") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

# Chuyển đổi DataFrame Pandas sang Spark DataFrame
df_HPG_spark = spark.createDataFrame(df_HPG)

df_HPG_spark.write.mode('overwrite').saveAsTable('df_HPG')

    from datetime import date, timedelta, datetime
# Tính toán ngày bắt đầu và kết thúc
    today = datetime.today()
    start_date = today - timedelta(days=3*365)  # 3 năm trước

    print("Ngày bắt đầu:", start_date.strftime('%d/%m/%Y'))
    print("Ngày kết thúc:", today.strftime('%d/%m/%Y'))

    df_HPG_sort = df_HPG[df_HPG['Date'] > start_date]

    # Dự đoán giá
    def Predict_future_prices_stocks(df):
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)
        df.set_index('Date', inplace=True)
        y = df['Price'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        y = scaler.fit_transform(y)

        n_lookback = 60
        n_forecast = 120
        X, Y = [], []

        for i in range(n_lookback, len(y) - n_forecast + 1):
            X.append(y[i - n_lookback: i])
            Y.append(y[i: i + n_forecast])

        X, Y = np.array(X), np.array(Y)

        # Fit the model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
        model.add(LSTM(units=50))
        model.add(Dense(n_forecast))

        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X, Y, epochs=100, batch_size=32, verbose=0)

        # Generate forecasts
        X_ = y[-n_lookback:].reshape(1, n_lookback, 1)
        Y_ = model.predict(X_).reshape(-1, 1)
        Y_ = scaler.inverse_transform(Y_)

        df_future = pd.DataFrame({
            'Date': pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=n_forecast),
            'Forecast': Y_.flatten()
        })

        return df_future

HPG_pre = Predict_future_prices_stocks(df_HPG_sort[['Date', 'Price']])

# Chuyển đổi DataFrame Pandas sang Spark DataFrame
HPG_pre_spark = spark.createDataFrame(HPG_pre)

HPG_pre_spark.write.mode('overwrite').saveAsTable('HPG_pre')    
