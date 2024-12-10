import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.src.layers import Dense, LSTM, Input
from keras.src.metrics import MeanSquaredError
from keras.src.models import Sequential
from keras.src.optimizers import SGD
from keras.src.optimizers.schedules import ExponentialDecay
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
from sklearn.preprocessing import MinMaxScaler

def preprocess(dataset: pd.DataFrame) -> (pd.DataFrame, MinMaxScaler):
    dataset = dataset[['가격', '최초등록일', '연식', '주행거리', '최대토크', '배기량', '최고출력', '연비']].copy()

    # parse date
    dataset['최초등록일'] = pd.to_datetime(dataset['최초등록일'], yearfirst=True, dayfirst=False)

    # set index with 최초등록일
    dataset = dataset.set_index('최초등록일')

    # transform data to proper float format
    # '연식' 컬럼을 문자열로 변환 (숫자가 포함된 경우 대비)
    dataset['연식'] = dataset['연식'].astype(str)
    # '연식' 데이터 정리 (숫자 뒤에 붙은 '.0' 제거)
    dataset['연식'] = dataset['연식'].apply(lambda x: x.split('.')[0] + '.' + x.split('.')[1][:2] if '.' in x else x)
    dataset['연식'] = pd.to_datetime(dataset['연식'], format='%Y.%m', errors='coerce')
    dataset['연식'] = dataset['연식'].apply(lambda x: x.timestamp() if pd.notnull(x) else None)

    dataset['연비'] = dataset['연비'].apply(lambda x: str(x).replace("km/ℓ",""))
    dataset['최대토크'] = dataset['최대토크'].apply(lambda x: str(x).replace("kg.m",""))
    dataset['최고출력'] = dataset['최고출력'].apply(lambda x: str(x).replace("마력",""))
    dataset['주행거리'] = pd.to_numeric(dataset['주행거리'], errors='coerce')
    dataset['최대토크'] = pd.to_numeric(dataset['최대토크'], errors='coerce')
    dataset['배기량'] = pd.to_numeric(dataset['배기량'], errors='coerce')
    dataset['최고출력'] = pd.to_numeric(dataset['최고출력'], errors='coerce')
    dataset['연비'] = pd.to_numeric(dataset['연비'], errors='coerce')

    # drop NaN, sort by datetime index and reset index
    print('Total NaN count:\n', dataset.isna().sum())
    dataset = dataset.dropna(axis=0, how='any')
    dataset = dataset.sort_index()
    # dataset = dataset.reset_index(drop=True)

    print('Preprocessed Dataset:\n', dataset)

    return dataset


def plot(true, predicted, divider, scaler):

    predict_plot = scaler.inverse_transform(predicted[0])
    true_plot = scaler.inverse_transform(true[0])

    predict_plot = predict_plot[:,0]
    true_plot = true_plot[:,0]

    plt.figure(figsize=(16,6))
    plt.plot(true_plot, label='True',linewidth=5)
    plt.plot(predict_plot,  label='Predict',color='y')

    if divider > 0:
        maxVal = max(true_plot.max(),predict_plot.max())
        minVal = min(true_plot.min(),predict_plot.min())

        plt.plot([divider,divider],[minVal,maxVal],label='train/test limit',color='k')

    plt.legend()
    plt.show()


def main3():
    # read dataset
    # https://stackoverflow.com/questions/50054419/extremely-poor-prediction-lstm-time-series
    data_name = 'genesis_large.csv'
    raw = pd.read_csv(f'dataset/{data_name}')
    print(raw.shape)

    # Preprocessing

    raw = preprocess(raw)

    scaler = MinMaxScaler()
    raw = scaler.fit_transform(raw)

    time_shift = 7  # shift is the number of steps we are predicting ahead
    n_rows = raw.shape[0]  # n_rows is the number of time steps of our sequence
    n_feats = raw.shape[1]
    train_size = int(n_rows * 0.8)

    # I couldn't understand how "ds" worked, so I simply removed it because in the code below it's not necessary

    # getting the train part of the sequence
    train_data = raw[:train_size, :]  # first train_size steps, all 5 features
    test_data = raw[train_size:, :]  # I'll use the beginning of the data as state adjuster

    # train_data = shuffle(train_data) !!!!!! we cannot shuffle time steps!!! we lose the sequence doing this

    x_train = train_data[:-time_shift, :]  # the entire train data, except the last shift steps
    x_test = test_data[:-time_shift, :]  # the entire test data, except the last shift steps
    x_predict = raw[:-time_shift, :]  # the entire raw data, except the last shift steps

    y_train = train_data[time_shift:, :]
    y_test = test_data[time_shift:, :]
    y_predict_true = raw[time_shift:, :]

    x_train = x_train.reshape(1, x_train.shape[0],
                              x_train.shape[1])  # ok shape (1,steps,5) - 1 sequence, many steps, 5 features
    y_train = y_train.reshape(1, y_train.shape[0], y_train.shape[1])
    x_test = x_test.reshape(1, x_test.shape[0], x_test.shape[1])
    y_test = y_test.reshape(1, y_test.shape[0], y_test.shape[1])
    x_predict = x_predict.reshape(1, x_predict.shape[0], x_predict.shape[1])
    y_predict_true = y_predict_true.reshape(1, y_predict_true.shape[0], y_predict_true.shape[1])

    print("\nx_train:")
    print(x_train.shape)
    print("y_train")
    print(y_train.shape)
    print("x_test")
    print(x_test.shape)
    print("y_test")
    print(y_test.shape)


    # Model

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(None, x_train.shape[2])))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(n_feats, return_sequences=True))

    model.compile(loss='mse', optimizer='adam')


    # Fitting

    # notice that I'm predicting from the ENTIRE sequence, including x_train
    # is important for the model to adjust its states before predicting the end
    model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=2, validation_data=(x_test, y_test))


    # Predicting

    y_predict_model = model.predict(x_predict)

    print("\ny_predict_true:")
    print(y_predict_true.shape)
    print("y_predict_model: ")
    print(y_predict_model.shape)

    test_size = n_rows - train_size
    print("test length: " + str(test_size))

    plot(y_predict_true, y_predict_model, train_size, scaler)
    plot(y_predict_true[:, -2 * test_size:], y_predict_model[:, -2 * test_size:], test_size, scaler)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main3()