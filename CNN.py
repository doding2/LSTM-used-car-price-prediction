import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Input
from keras.src.models import Sequential, Model
from keras.src.optimizers import Adam
from keras.src.optimizers.schedules import ExponentialDecay
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
from sklearn.preprocessing import MinMaxScaler

def preprocess_with_no_scaling(dataset: pd.DataFrame) -> pd.DataFrame:
    # '이름' 컬럼이 '기아 K5'로 시작하는 데이터만 필터링
    # dataset = dataset[dataset['이름'].astype(str).str.startswith('기아 K7')]

    dataset = dataset[['신차대비가격', '최초등록일', '연식', '주행거리', '최대토크', '배기량', '최고출력', '연비']].copy()

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
    dataset['신차대비가격'] = pd.to_numeric(dataset['신차대비가격'], errors='coerce')
    dataset = dataset[dataset['신차대비가격'] > 0]

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
    dataset = dataset.copy()
    ts_train = dataset[:'2023'].iloc[:, 0:1].values
    ts_test = dataset['2024':].iloc[:, 0:1].values
    scaler = MinMaxScaler()
    ts_train_scaled = scaler.fit_transform(ts_train)
    X_train, y_train = [], []
    for i in range(time_steps, len(ts_train) - 1):
        X_train.append(ts_train_scaled[i - time_steps:i, 0])
        y_train.append(ts_train_scaled[i:i + for_periods, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    inputs = dataset['신차대비가격'].values[len(dataset) - len(ts_test) - time_steps:].reshape(-1, 1)
    inputs = scaler.transform(inputs)
    X_test = np.array([inputs[i - time_steps:i, 0] for i in range(time_steps, len(ts_test) + time_steps - for_periods)])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return X_train, y_train, X_test, scaler

def CNN_model(X_train, y_train, X_test, scaler, loss_function='mean_squared_error'):
    time_steps = X_train.shape[1]  # X_train.shape = (batch, time_steps, 1)
    kernel_size = min(3, time_steps)  # kernel_size가 time_steps보다 크지 않도록 설정

    model = Sequential([
        Conv1D(filters=64, kernel_size=kernel_size, activation='relu', input_shape=(time_steps, 1)),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=kernel_size, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1)
    ])

    lr_schedule = ExponentialDecay(initial_learning_rate=0.01, decay_steps=100000, decay_rate=0.96, staircase=False)
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss=loss_function)

    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=128, verbose=0, callbacks=[early_stopping])

    CNN_prediction = scaler.inverse_transform(model.predict(X_test))
    return model, CNN_prediction


def confirm_result(y_test, y_pred):
    MAE, RMSE = mean_absolute_error(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred))
    RMSLE, R2 = np.sqrt(mean_squared_log_error(y_test, y_pred)), r2_score(y_test, y_pred)
    return pd.DataFrame({'Results': [MAE, RMSE, RMSLE, R2]}, index=['MAE', 'RMSE', 'RMSLE', 'R2'])

def plot_prediction(all_data, y_pred):
    actual_pred = pd.DataFrame({'Real': all_data.loc['2024':, '신차대비가격'][:len(y_pred)], 'Predict': y_pred[:, 0]})
    return actual_pred.plot()

def main4():
    dataset = pd.read_csv('dataset/chevrolet_daewoo_compact.csv')
    dataset = preprocess_with_no_scaling(dataset)
    X_train, y_train, X_test, scaler = prepare_train_test_normalize(dataset, 10, 2)
    model, CNN_prediction = CNN_model(X_train, y_train, X_test, scaler)
    y_pred = pd.DataFrame(CNN_prediction[:, 0])
    y_test = dataset.loc['2024':, '신차대비가격'][:len(CNN_prediction)].reset_index(drop=True)
    evaluation_results = confirm_result(y_test, y_pred)
    print(evaluation_results)
    plot_prediction(dataset, CNN_prediction)
    plt.gca().text(0.02, 0.98, f"MAE: {evaluation_results['Results']['MAE']:.4f}\nRMSE: {evaluation_results['Results']['RMSE']:.4f}\nRMSLE: {evaluation_results['Results']['RMSLE']:.4f}\nR²: {evaluation_results['Results']['R2']:.4f}",
                   transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    plt.show()

if __name__ == '__main__':
    main4()
