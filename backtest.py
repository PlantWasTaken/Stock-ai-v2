import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
from keras import layers
from keras.models import load_model
from rich.progress import track
import plotly.express as px

SEQ_LEN = 60


from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv(r'E:\ai\stock\stock\v2.1\simpledata.csv')['Close']
scalar = MinMaxScaler()
scaled_data = scalar.fit_transform(df.values.reshape(-1,1))

# Load the saved model
loaded_model = tf.keras.models.load_model(r"E:\ai\stock\stock\v2.1\simple21.h5")





# Number of future time steps to predict
future_steps = 300

ft = np.array([scaled_data[i:i+SEQ_LEN] for i in range(future_steps-SEQ_LEN)])
print(ft.shape)

Y3 = loaded_model.predict(ft)

Y2 = df[0+SEQ_LEN:len(Y3)+SEQ_LEN]
predicted_prices = np.squeeze(scalar.inverse_transform(np.array(Y3)))
#predicted_prices = np.squeeze(Y3)
print(predicted_prices)

fig3 = px.line(y=[Y2], title='Stonk')
fig2 =  px.line(y=[predicted_prices,Y2], title='Stonk')
fig = px.line(y=[predicted_prices], title='Stonk')

fig3.show()
fig2.show()
fig.show()
exit()



fig = px.line(y=[Y2], title='Stonk')
fig3 = px.line(y=[Y3], title='Stonk')


fig.show()
