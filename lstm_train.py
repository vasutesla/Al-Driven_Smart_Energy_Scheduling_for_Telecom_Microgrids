import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense
#from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error


def create_sequence(data, window_size):
    x=[]
    y=[]
    for i in range(window_size, len(data)):
        x.append(data[i-window_size : i])
        y.append(data[i])

    return np.array(x),np.array(y)


tower_hourly=pd.read_csv("tower_hourly_MT_321.csv")
data=tower_hourly["energy_demand_kWh"].values.reshape(-1,1)

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(data)

window_size=24
x,y=create_sequence(scaled_data,window_size)

train_size=int(0.8*len(x))
x_train=x[:train_size]
x_test=x[train_size:]

y_train=y[:train_size]
y_test=y[train_size:]

model=Sequential()
model.add(LSTM(50,activation='tanh',input_shape=(window_size,1)))
#model.add(Dense(25,activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mean_squared_error')
model.summary()

#early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history=model.fit(x_train,y_train,epochs=20, batch_size=32,validation_data=(x_test,y_test),verbose=1)

predictions=model.predict(x_test)
predictions=scaler.inverse_transform(predictions)
y_test_actual=scaler.inverse_transform(y_test)


mae=mean_absolute_error(y_test_actual,predictions)
rmse=np.sqrt(mean_squared_error(y_test_actual,predictions))
mape=np.mean(np.abs((y_test_actual-predictions)/y_test_actual))*100

model.save("lstm_model_new.h5")
joblib.dump(scaler, "scaler.save")

print("LSTM mae : ",mae)
print("LSTM rmse : ",rmse)
print("LSTM mape : ",mape)


plt.figure()
plt.plot(y_test_actual[:48],label="Actual")
plt.plot(predictions[:48],label="LSTM predictions")
plt.legend()
plt.title("LSTM:Actual vs Predicted")
plt.show()