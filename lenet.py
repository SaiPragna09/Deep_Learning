import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


# -------------------------------
# 1. Load and Preprocess Dataset
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
# 2. Build LeNet-5 Model
# -------------------------------
model = Sequential()

# C1 Layer
model.add(Conv2D(filters=6, kernel_size=(5,5), activation='tanh', input_shape=(28,28,1)))

# S2 Layer
model.add(AveragePooling2D(pool_size=(2,2)))

# C3 Layer
model.add(Conv2D(filters=16, kernel_size=(5,5), activation='tanh'))

# S4 Layer
model.add(AveragePooling2D(pool_size=(2,2)))

# Flatten
model.add(Flatten())

# Fully Connected Layers
model.add(Dense(120, activation='tanh'))
model.add(Dense(84, activation='tanh'))
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


# -------------------------------
# 7. Visualize Filters
# -------------------------------
filters = model.layers[0].get_weights()[0]

plt.figure(figsize=(8,6))

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(filters[:,:,0,i], cmap='gray')
    plt.axis('off')

plt.suptitle("First Convolution Layer Filters")
plt.show()