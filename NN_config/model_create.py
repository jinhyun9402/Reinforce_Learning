from keras.optimizers import Adam
from keras import models
from keras import layers

input_tensor = layers.Input(shape=(4, ))
x = layers.Dense(40, activation='tanh', kernel_initializer='he_uniform')(input_tensor)
x = layers.Dense(40, activation='tanh', kernel_initializer='he_uniform')(x)
x = layers.Dense(40, activation='tanh', kernel_initializer='he_uniform')(x)
x = layers.Dense(40, activation='tanh', kernel_initializer='he_uniform')(x)
output_tensor = layers.Dense(6, activation='relu', kernel_initializer='he_uniform')(x)

model = models.Model(inputs=input_tensor, outputs=output_tensor)
model.summary()
model.compile(loss='mse', optimizer=Adam(lr=0.001))

model_json = model.to_json()
with open('./save_model/controller_model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('./save_model/cont_weifht.h5')