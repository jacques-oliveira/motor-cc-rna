#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 10:28:50 2024

@author: jacques
"""
import pandas as pd
bancoDB = pd.read_csv('./BD_Jean_98000.csv')
print(bancoDB.head())

#%%
print(bancoDB.isna().sum())
#%%
#Separando entradas e saída
entradas = bancoDB[['T','w','i']].values
saida = bancoDB['V'].values
#%%
print(entradas)
#%%
print(saida)
#%%
# Dividir os dados em conjuntos de treinamento e teste (80% treino, 20% teste)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(entradas, saida, test_size=0.2, random_state=42)

#%%
# Normalização dos dados
from sklearn.preprocessing import StandardScaler , MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%%
#Criar modelo
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

motorccRNA = Sequential()
motorccRNA.add(Dense(64, input_dim=3, activation='relu'))
motorccRNA.add(Dense(32, activation='relu'))
motorccRNA.add(Dense(1, activation='linear'))

#%%#Compilar o modelo
from keras.optimizers import Adam
optmizer = Adam(learning_rate=0.001)
motorccRNA.compile(optimizer = optmizer, loss='mean_squared_error',metrics=['mean_absolute_error'])
#%%
#Compilar o modelo
from keras.optimizers import Adam
optmizer = Adam(learning_rate=0.001)
motorccRNA.compile(optimizer = optmizer, loss='mean_squared_error',metrics=['mean_absolute_error'])
#%%
#Treinar
motorccRNA.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))
#%%
print(saida.mean())

#%%
previsao = motorccRNA.predict(X_test)

#%%

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, previsao)
print(mae)

#%%