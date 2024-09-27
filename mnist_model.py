from data_preprocessing import data_loader as dl
import numpy as np
import pandas as pd
import keras as kr
import matplotlib.pyplot as plt

def init_model():
    model = kr.models.Sequential([
        kr.layers.Input(shape=(28,28)),
        kr.layers.Flatten(),
        kr.layers.Dense(units=32, activation=kr.activations.relu),
        kr.layers.Dense(units=10, activation=kr.activations.softmax),
    ], name='AKIMBo') #* AI - Keras - Intelligent - Mnist - Bot 
    model.compile(
        optimizer=kr.optimizers.Adam(learning_rate=0.01),
        loss=kr.losses.SparseCategoricalCrossentropy(),
        metrics=[kr.metrics.SparseCategoricalAccuracy(name='accuracy')],
    )
    model.summary()
    mnist_dataset = dl.load_mnist()
    history = model.fit(
        x=mnist_dataset['x_train'], 
        y=mnist_dataset['y_train'],
        batch_size=30,
        epochs=20,
        validation_split=0.2,
        verbose=1
    )
    metrics = model.evaluate(
        x=mnist_dataset['x_test'],
        y=mnist_dataset['y_test'],
        batch_size=30,
        verbose=1
    )
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['loss'])
    plt.legend(['accuracy', 'loss'])
    plt.title(f'AKIMBo metrics {metrics}')
    plt.xlabel('epoch')
    plt.show()
    model.save('./AKIMBo.keras')

def load_model(path:str) -> kr.models.Model:
    return kr.models.load_model(path)

model = load_model('./AKIMBo.keras')
model.summary()
    
def predict_digit(model:kr.models.Model, path:str, digit:int) -> None:
    print('Predictions for digit: ', digit)
    img = np.array([dl.load_image(path)])
    predictions = model.predict(x=img, verbose=1)
    for i in range(predictions[0].size):
        print('\tDigit: ', i, '\tConfidence:', round(predictions[0][i]*100, 3), '%')
        
predict_digit(model, '.ignored_files/0.png', 0)
predict_digit(model, '.ignored_files/1.png', 1)
predict_digit(model, '.ignored_files/2.png', 2)
predict_digit(model, '.ignored_files/5.png', 5)
predict_digit(model, '.ignored_files/8.png', 8)
predict_digit(model, '.ignored_files/9.png', 9)
predict_digit(model, '.ignored_files/9.2.png', 9)
    