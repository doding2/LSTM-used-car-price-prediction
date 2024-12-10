import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Dense, LSTM, Input, Embedding, Bidirectional
from keras.src.metrics import MeanSquaredError
from keras.src.models import Sequential
from keras.src.optimizers import SGD, Adam
from keras.src.optimizers.schedules import ExponentialDecay
from keras.src.layers import Attention, Concatenate
from keras.src import Model
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import Dropout

from 추천코드_keras import GradientEnhancedLoss, AsymmetricLoss


def preprocess_with_no_scaling(dataset: pd.DataFrame) -> pd.DataFrame:
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

    # # min-max scaling
    # scaler = MinMaxScaler()
    # dataset = pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns, index=dataset.index)
    print('Preprocessed Dataset:\n', dataset)

    return dataset


def prepare_train_test_normalize(dataset: pd.DataFrame, time_steps, for_periods):
    # training, test set 만들기
    dataset = dataset.copy()
    ts_train = dataset[:'2023'].iloc[:, 0:1].values
    ts_test = dataset['2024':].iloc[:, 0:1].values
    ts_train_len = len(ts_train)
    ts_test_len = len(ts_test)

    # 데이터 scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    ts_train_scaled = scaler.fit_transform(ts_train)

    # training 데이터의 samples와 time steps로 원본 데이터 슬라이싱 하기
    X_train = []
    y_train = []
    for i in range(time_steps, ts_train_len - 1):
        X_train.append(ts_train_scaled[i-time_steps:i, 0])
        y_train.append(ts_train_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # 3차원으로 재구성 하기
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # X_test 만들 준비 하기
    inputs = pd.concat((dataset['가격'][:'2023'], dataset['가격']['2024':]), axis=0).values
    inputs = inputs[len(inputs) - len(ts_test) - time_steps:]
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(time_steps, ts_test_len + time_steps - for_periods):
        X_test.append(inputs[i-time_steps:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_test, scaler


def LSTM_model(X_train, y_train, X_test, scaler, loss_function='mean_squared_error'):
    # LSTM 아키텍쳐
    model = Sequential()
    model.add(Bidirectional(LSTM(units=256, return_sequences=True, input_shape=(X_train.shape[1], 1))))
    model.add(Bidirectional(LSTM(units=128, return_sequences=False)))
    model.add(Dense(units=1))

    # 컴파일링
    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=False
    )
    # model.compile(optimizer=SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=False), loss='mean_squared_error')
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss=loss_function)

    # training data 세트에 피팅하기
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=128, verbose=0, callbacks=[early_stopping])

    # X_test를 모델에 넣어서 예측하기
    LSTM_prediction = model.predict(X_test)

    # 스케일러에 예측값 넣어 반환하기
    LSTM_prediction = scaler.inverse_transform(LSTM_prediction)

    return model, LSTM_prediction

def LSTM_model_with_attention(X_train, y_train, X_test, scaler, loss_function='mean_squared_error'):
    # Input 정의
    input_layer = Input(shape=(X_train.shape[1], 1))

    # LSTM 레이어
    lstm_out = Bidirectional(LSTM(units=256, return_sequences=True))(input_layer)
    lstm_out = Bidirectional(LSTM(units=128, return_sequences=True))(lstm_out)

    # Attention 레이어 추가
    attention = Attention()([lstm_out, lstm_out])  # Self-Attention: Query = Key = Value = lstm_out

    # Attention 결과를 Dense 레이어와 결합
    concat = Concatenate()([lstm_out, attention])
    dense_out = Dense(units=1)(concat)

    # 모델 정의
    model = Model(inputs=input_layer, outputs=dense_out)

    # 컴파일
    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=False
    )
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss=loss_function)

    # 모델 훈련
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=128, verbose=0, callbacks=[early_stopping])

    # 예측 수행
    LSTM_prediction = model.predict(X_test)
    LSTM_prediction = np.reshape(LSTM_prediction, (LSTM_prediction.shape[0], -1))
    LSTM_prediction = scaler.inverse_transform(LSTM_prediction)

    return model, LSTM_prediction


def LSTM_model_bidirectional(X_train, y_train, X_test, scaler, loss_function='mean_squared_error'):
    # LSTM 아키텍쳐
    model = Sequential()
    model.add(Bidirectional(LSTM(units=256, return_sequences=True, input_shape=(X_train.shape[1], 1))))
    model.add(Bidirectional(LSTM(units=128, return_sequences=False)))
    model.add(Dense(units=1))

    # 컴파일링
    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=False
    )
    # model.compile(optimizer=SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=False), loss='mean_squared_error')
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss=loss_function)

    # training data 세트에 피팅하기
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=128, verbose=0, callbacks=[early_stopping])

    # X_test를 모델에 넣어서 예측하기
    LSTM_prediction = model.predict(X_test)

    # 스케일러에 예측값 넣어 반환하기
    LSTM_prediction = scaler.inverse_transform(LSTM_prediction)

    return model, LSTM_prediction


def plot_prediction(all_data, y_pred):
    actual_pred = pd.DataFrame(columns=['Real', 'Predict'])
    actual_pred['Real'] = all_data.loc['2024':, '가격'][0:len(y_pred)]
    actual_pred['Predict'] = y_pred[:, 0]

    m = MeanSquaredError()
    m.update_state(np.array(actual_pred['Real']), np.array(actual_pred['Predict']))

    return m.result().numpy(), actual_pred.plot()


def confirm_result(y_test, y_pred):
    MAE = mean_absolute_error(y_test, y_pred)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    MSLE = mean_squared_log_error(y_test, y_pred)
    RMSLE = np.sqrt(mean_squared_log_error(y_test, y_pred))
    R2 = r2_score(y_test, y_pred)

    pd.options.display.float_format = '{:.5f}'.format
    Result = pd.DataFrame(data=[MAE, RMSE, RMSLE, R2],
                          index=['MAE', 'RMSE', 'RMSLE', 'R2'],
                          columns=['Results'])
    return Result


def main2():
    # read dataset
    data_name = 'kia_large.csv'
    dataset = pd.read_csv(f'dataset/{data_name}')

    # preprocessing
    dataset = preprocess_with_no_scaling(dataset)

    # split train and test with normalization
    X_train, y_train, X_test, scaler = prepare_train_test_normalize(dataset, 5, 2)

    # predict using LSTM
    # 3.a GradientEnhancedLoss 적용
    # custom_loss = GradientEnhancedLoss(alpha=0.7, beta=1.5)

    # 3.b AsymmetricLoss 적용
    # custom_loss = AsymmetricLoss(threshold=0.1, penalty_factor=2.0)

    model, LSTM_prediction = LSTM_model_with_attention(X_train, y_train, X_test, scaler)

    # evaluate prediction performance
    y_pred = pd.DataFrame(LSTM_prediction[:, 0])
    y_test = dataset.loc['2024':, '가격'][0:len(LSTM_prediction)]
    y_test.reset_index(drop=True, inplace=True)
    evaluation_results = confirm_result(y_test, y_pred)
    print(evaluation_results)

    # plot prediction and performance
    plot_prediction(dataset, LSTM_prediction)
    metrics_text = (
        f"MAE: {evaluation_results['Results']['MAE']:.4f}\n"
        f"RMSE: {evaluation_results['Results']['RMSE']:.4f}\n"
        f"RMSLE: {evaluation_results['Results']['RMSLE']:.4f}\n"
        f"R²: {evaluation_results['Results']['R2']:.4f}"
    )
    plt.gca().text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes,
                   fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main2()