import numpy as np
import pandas as pd
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

dataframe = pd.read_csv("D:/#2024/MBE/2015-2019_2024.csv")
cols = list(dataframe)
print(cols)
X = dataframe.values[0:, 0:-1]
print(X)
y = dataframe.values[0:, -1]
print(X.shape)
count = 1
kf = KFold(n_splits=10)
for train, test in kf.split(X):
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]
    print(X_train)
    print(y_train)
    original = X
    transformer = MinMaxScaler().fit(X)  # fit does nothing
    X = transformer.transform(X)
    input_feature = Input( shape=(36,) )
    h1 = Dense(18, activation='relu')(input_feature )
    h2 = Dense(9, activation='relu')(h1)
    encoded = Dense( 1, activation='relu')( h2)
    h3 = Dense(9, activation='tanh' )(encoded)
    h4 = Dense(18, activation='tanh')(h3)
    decoded = Dense(36, activation='tanh')(h4)
    autoencoder = Model( input_feature, decoded )
    encoder = Model( input_feature, encoded )
    autoencoder.compile( optimizer='adam', loss='mean_squared_error', metrics=['mae'] )
    es1 = EarlyStopping( monitor='val_loss', mode='min', verbose=1, patience=200 )
    mc1 = ModelCheckpoint('D:/#2024/MBE/ae_2024-1.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    autoencoder.fit(X, X, epochs=1000, validation_data=(X, X), verbose=2, callbacks=[es1, mc1] )
    autoencoder.save('D:/#2024/MBE/ae_2024-1.h5')
    encoder.save('D:/#2024/MBE/ae_2024-1.h5')
    print('Hello')
    standard_ae_model = load_model('D:/#2024/MBE/ae_2024-1.h5')
    x_decoded = standard_ae_model.predict(X)
    x_error = np.mean(np.power(X - x_decoded, 2), axis=1)
    average = np.average(x_error)
    std = np.std(x_error)
    th1 = average + std
    thresh = average + std
#   thresh = np.percentile(th1, 25)
#   thresh = np.percentile(th1, 50)
    thresh = np.percentile(th1, 75)
    print('thresh=', thresh)
    error_frame = pd.DataFrame(x_error, columns=['re'])
    X_bool = error_frame['re'] <= thresh
    clean_Data = dataframe[X_bool]
    clean_Data.to_csv('D:/#2024/MBE/clean_all.csv', sep=',', encoding='utf-8', index=False)
    print('finish')