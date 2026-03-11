import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical


# -------------------------------
# 1. Load Dataset
# -------------------------------
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# Reshape for CNN
train_x = train_x.reshape(-1, 28, 28, 1)
test_x = test_x.reshape(-1, 28, 28, 1)

# Normalize values
train_x = train_x / 255.0
test_x = test_x / 255.0

# One-hot encoding
train_y = to_categorical(train_y, 10)
test_y = to_categorical(test_y, 10)


# -------------------------------
# 2. Build AlexNet Model
# -------------------------------
model = Sequential()

model.add(Conv2D(96, (5,5), strides=1, activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(384, (3,3), padding='same', activation='relu'))
model.add(Conv2D(384, (3,3), padding='same', activation='relu'))
model.add(Conv2D(256, (3,3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))


# -------------------------------
# 3. Compile Model
# -------------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# -------------------------------
# 4. Model Summary
# -------------------------------
model.summary()


# -------------------------------
# 5. Train Model
# -------------------------------
model.fit(
    train_x,
    train_y,
    epochs=1,
    batch_size=128,
    validation_data=(test_x, test_y)
)


# -------------------------------
# 6. Evaluate Model
# -------------------------------
loss, accuracy = model.evaluate(test_x, test_y)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)