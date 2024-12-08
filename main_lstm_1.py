import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Dense, LSTM, Dropout
from keras.src.models import Sequential
from keras.src.optimizers import Adam
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def preprocess(dataset: pd.DataFrame) -> pd.DataFrame:
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


def group_by_kmeans_clustering(dataset: pd.DataFrame, data_name: str = '') -> dict[str, pd.Series]:
    pca = PCA(n_components=2)
    pca.fit(dataset)
    pca_data = pd.DataFrame(data = pca.transform(dataset), columns=['pc1', 'pc2'])

    x = []
    y = []

    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, random_state=7, n_init=10)
        kmeans.fit(pca_data)

        x.append(k)
        y.append(kmeans.inertia_)

    plt.plot(x, y)
    plt.title(f'Elbow Method\n({data_name})')
    plt.show()

    # optimal k is 4 with genesis_large data
    # optimal k is 3 with hyundai_large data
    k = 4
    model = KMeans(n_clusters=k, random_state=7, n_init=10)
    model.fit(pca_data)
    pca_data['labels'] = model.predict(pca_data)
    dataset['labels'] = pca_data['labels']
    sns.scatterplot(x='pc1', y='pc2', hue='labels', data=pca_data)
    plt.title(f'Cluster results\n({data_name})')
    plt.show()

    # group result by labels
    clusters = dataset['labels'].unique()
    results = {cluster: dataset[dataset['labels'] == cluster].drop(columns='labels') for cluster in clusters}


    return results


def group_by_dbscan_clustering(dataset: pd.DataFrame, data_name: str = '') -> dict[str, pd.DataFrame]:
    # PCA를 사용해 차원 축소
    pca = PCA(n_components=2)
    pca.fit(dataset)
    pca_data = pd.DataFrame(data=pca.transform(dataset), columns=['pc1', 'pc2'])

    # DBSCAN 클러스터링
    eps = 0.07  # 두 샘플 간 최대 거리
    min_samples = 5  # 클러스터 형성을 위한 최소 샘플 수
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(pca_data)

    # 클러스터 결과 시각화
    pca_data['labels'] = labels
    dataset['labels'] = labels

    sns.scatterplot(x='pc1', y='pc2', hue='labels', data=pca_data, palette='viridis', legend="full")
    plt.title(f'DBSCAN Cluster Results\n({data_name})')
    plt.show()

    # 노이즈(-1로 라벨링된 데이터)는 제외하고 클러스터별로 그룹화
    clusters = [label for label in set(labels) if label != -1]
    results = {cluster: dataset[dataset['labels'] == cluster].drop(columns='labels') for cluster in clusters}

    print(f"DBSCAN: {len(clusters)} clusters formed (excluding noise).")
    print(f"Noise samples: {sum(labels == -1)}")

    return results

def predict_with_lstm(dataset: pd.DataFrame, cluster_label: str, data_name: str = ''):
    dataset = dataset.copy()
    dataset_X = dataset[['연식', '주행거리', '최대토크', '배기량', '최고출력', '연비']].to_numpy()
    dataset_y = dataset[['가격']].to_numpy()

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
    if len(X) == 0:
        return

    X = np.array(X)
    y = np.array(y)

    # 테스트 데이터의 개수가 window size만큼 있도록 조정
    if len(X) * 0.3 < window_size:
        test_size = 0.3
    else:
        test_size = window_size / len(X)
    print('test_size: ', test_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)

    # reshape y for LSTM compatibility
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    print('훈련 데이터의 크기 :', X_train.shape, y_train.shape)
    print('테스트 데이터의 크기 :', X_test.shape, y_test.shape)

    # prepare hyperparameters of model
    # 시계열 데이터에서 활성화 함수는 relu보다 tanh가 나음
    model = Sequential()
    model.add(LSTM(units=50, activation='tanh', return_sequences=True, input_shape=(window_size, 6)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, activation='tanh', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.summary()

    # train model
    # 초기 학습률 (learning_rate) 0.001에서 0.0005로 조정
    # 과적합을 방지하고 훈련 시간을 단축하기 위해 EarlyStopping 콜백을 활용
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, callbacks=[early_stopping])
    y_pred = model.predict(X_test)

    # 성능 평가 (스케일링 복구 포함)
    scaler = MinMaxScaler()
    scaler.fit(dataset[['가격']])
    evaluation_results = evaluate_model(y_test, y_pred, scaler)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(scaler.inverse_transform(y_test.reshape(-1, 1)), color='red', label='Real Price')
    plt.plot(scaler.inverse_transform(y_pred.reshape(-1, 1)), color='blue', label='Predicted Price')

    # Add evaluation metrics to the plot
    metrics_text = (
        f"RMSE: {evaluation_results['RMSE']:.4f}\n"
        f"MAE: {evaluation_results['MAE']:.4f}\n"
        f"R²: {evaluation_results['R²']:.4f}\n"
        f"MAPE: {evaluation_results['MAPE']:.2f}%"
    )
    plt.gca().text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes,
                   fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    plt.title(f'Prediction of Cluster {cluster_label}\n({data_name})')
    plt.xlabel('Index')
    plt.ylabel('Used Car Price')
    plt.legend()
    plt.show()


def evaluate_model(y_test, y_pred, scaler):
    # 역변환 (스케일링 복구)
    # y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    # y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test = y_test.flatten()
    y_pred = y_pred.flatten()

    # RMSE 계산
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # MAE 계산
    mae = mean_absolute_error(y_test, y_pred)
    # R² 계산
    r2 = r2_score(y_test, y_pred)
    # MAPE 계산
    mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1e-10))) * 100  # Zero division protection

    # 결과 출력
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")

    return {"RMSE": rmse, "MAE": mae, "R²": r2, "MAPE": mape}


def main():
    # read dataset
    data_name = 'kia_large.csv'
    dataset = pd.read_csv(f'dataset/{data_name}')

    # 특정 모델만 걸러내보기
    # dataset = dataset.loc[dataset['이름'].str.startswith('기아 K5')]
    # print(dataset)

    # preprocessing
    dataset = preprocess(dataset)

    # group data by clustering
    dataset_dict = group_by_dbscan_clustering(dataset, data_name)

    # check result by printing
    for cluster, df_cluster in dataset_dict.items():
        print(f"Count of Cluster {cluster}: {len(df_cluster)}\n")
        # predict_with_lstm2(pd.DataFrame(df_cluster), cluster, data_name)

    predict_with_lstm(dataset, "All", data_name)

if __name__ == '__main__':
    main()

