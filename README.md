# Fashion MNIST
- Machine learning project using supervised learning technique that implemented in Python programming language. 
- Programming language used: Python.

![MachineLearning](https://user-images.githubusercontent.com/33843231/71016941-0ed91800-2120-11ea-8bbf-dd5f029467fe.jpg)

```
from tkinter import *
from tkinter import filedialog
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.utils import np_utils

fashion = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
           'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

clf = Sequential([
    Conv2D(filters=32, kernel_size=2, activation='relu', input_shape=(28, 28, 1)),
    Conv2D(filters=64, kernel_size=2, activation='relu'),
    MaxPooling2D(pool_size=2),
    Dropout(0.10),

    Flatten(),
    Dense(16, activation='relu'),
    Dropout(0.1),
    Dense(10, activation='softmax')
])

root = Tk()


def train_data():
    train_df = pd.read_csv('C:\\Users\\SHAHIN\\Desktop\\fashion-mnist_train.csv')
    data = np.array(train_df, dtype='float32')
    X_train = data[:, 1:]
    y_train = data[:, 0]

    X_train = X_train.astype('float32') / 255
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    y_train = np_utils.to_categorical(y_train, 10)

    clf.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr=0.001),
        metrics=['accuracy']
    )

    clf.fit(
        X_train, y_train, batch_size=512,
        epochs=2, verbose=1
    )


def test_data():
    test_data = np.array(pd.read_csv('C:\\Users\\SHAHIN\\Desktop\\fashion-mnist_test.csv'), dtype='float32')

    predict_value = int(input.get("1.0", "end-1c"))

    print("Predict Value: ", predict_value)

    X_test = test_data[predict_value, 1:] / 255
    y_test = test_data[predict_value, 0]

    X_test = X_test.astype('float32')
    X_test = X_test.reshape(1, 28, 28, 1)

    prediction = clf.predict(X_test)
    prediction = np.argmax(prediction)
    print("Prediction: ",prediction)

    detect = str(fashion[prediction])

    output.insert(INSERT, detect)
    ytest = int(y_test)
    print("Y_Test: ", fashion[ytest])


frame = Frame(root, width=600, height=500, bg="skyblue")

label = Label(frame, text="Fashin MNIST")
label.config(font=('Times New Roman', 12))
label.place(x=270, y=0)

train = Button(frame, text="Train", fg="white", bg="green", width=10, command=train_data)
train.config(font=('Times New Roman', 12))
train.place(x=270, y=50)

label = Label(frame, text="Input")
label.config(font=('Times New Roman', 12))
label.place(x=150, y=140)

label = Label(frame, text="(Enter any number from 0 to 9999)")
label.config(font=('Times New Roman', 8))
label.place(x=150, y=170)

input = Text(frame, height=1, width=15, fg='blue')
input.config(font=('Times New Roman', 14))
input.place(x=150, y=200)

test = Button(frame, text="Test", fg="white", bg="blue", width=10, command=test_data)
test.config(font=('Times New Roman', 12))
test.place(x=350, y=200)

label = Label(frame, text="Output")
label.config(font=('Times New Roman', 12))
label.place(x=200, y=300)

output = Text(frame, height=1, width=15, fg='blue')
output.config(font=('Times New Roman', 14))
output.place(x=280, y=300)

frame.pack()

root.mainloop()


```
