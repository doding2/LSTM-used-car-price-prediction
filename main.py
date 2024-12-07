import numpy as np
import pandas as pd
from scipy.constants import yocto
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
from keras.src.models import Sequential
from keras.src.layers import Dense, LSTM, Dropout
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def preprocessing(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset = dataset[['가격', '신차대비가격', '최초등록일', '조회수', '배기량', '색상', '연비', '최고출력', '최대토크']]

    # parse date
    dataset['최초등록일'] = pd.to_datetime(dataset['최초등록일'], yearfirst=True, dayfirst=False)

    # set index with 최초등록일
    dataset = dataset.set_index('최초등록일')

    # remove prefix and suffix from data
    dataset['연비'] = dataset['연비'].apply(lambda x: str(x).replace("km/ℓ",""))
    dataset['최고출력'] = dataset['최고출력'].apply(lambda x: str(x).replace("마력",""))
    dataset['최대토크'] = dataset['최대토크'].apply(lambda x: str(x).replace("kg.m",""))
    dataset['연비'] = pd.to_numeric(dataset['연비'], errors='coerce')
    dataset['최고출력'] = pd.to_numeric(dataset['최고출력'], errors='coerce')
    dataset['최대토크'] = pd.to_numeric(dataset['최대토크'], errors='coerce')

    # encode categorical data to numeric data
    encoder = LabelEncoder()
    dataset['색상'] = encoder.fit_transform(dataset['색상'])

    # min-max scaling and sort by date
    print(dataset.isna().sum())
    dataset = dataset.dropna(axis=0, how='any')
    dataset = dataset.sort_index()
    dataset = pd.DataFrame(MinMaxScaler().fit_transform(dataset), columns=dataset.columns, index=dataset.index)
    print(dataset)

    return dataset


def main():
    # read dataset
    dataset = pd.read_csv('dataset/genesis_large.csv')

    # preprocessing
    dataset = preprocessing(dataset)

    dataset_X = dataset[['신차대비가격', '조회수', '배기량', '색상', '연비', '최고출력', '최대토크']].values.tolist()
    dataset_y = dataset[['가격']].values.tolist()

    # group 30 data into a single window
    # and use these windows as X and y
    window_size = 30

    X = []
    y = []
    for i in range(len(dataset_y) - window_size):
        _X = dataset_X[i: i + window_size]
        _y = dataset_y[i + window_size]
        X.append(_X)
        y.append(_y)

    # devide dataset into train/valid/test
    print('전체 데이터의 크기 :', len(X), len(y))

    train_size = int(len(y) * 0.7)
    X_train = np.array(X[0: train_size])
    y_train = np.array(y[0: train_size])

    X_test = np.array(X[train_size: len(X)])
    y_test = np.array(y[train_size: len(y)])

    print('훈련 데이터의 크기 :', X_train.shape, y_train.shape)
    print('테스트 데이터의 크기 :', X_test.shape, y_test.shape)

    # prepare hyperparameters of model
    model = Sequential()
    model.add(LSTM(units=20, activation='relu', return_sequences=True, input_shape=(window_size, 7)))
    model.add(Dropout(0.1))
    model.add(LSTM(units=20, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=1))
    model.summary()

    # train model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=70, batch_size=30)
    y_pred = model.predict(X_test)

    # plot the results
    plt.figure()
    plt.plot(y_test, color='red', label='real price')
    plt.plot(y_pred, color='blue', label='predicted price')
    plt.title('used car price prediction')
    plt.xlabel('time')
    plt.ylabel('used car price')
    plt.legend()
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

