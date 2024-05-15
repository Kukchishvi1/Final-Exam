import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Function to preprocess images
def preprocess_image(image):
    image = tf.image.resize(image, (32, 32))  # Resize image to 32x32
    image = tf.cast(image, tf.float32) / 255.0  # Normalize pixel values
    return image

# Function to load gun images
def load_images(folder, num_images):
    images = []
    for i in range(1, num_images + 1):
        image = tf.keras.preprocessing.image.load_img(f'{folder}/gun{i}.jpg', target_size=(32, 32))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = preprocess_image(image)
        images.append(image)
    return np.array(images)

# Function to plot images
def plot_images(images, labels):
    plt.figure(figsize=(10, 5))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.title(labels[i])
        plt.axis('off')
    plt.show()

# Load CIFAR-10 dataset
(x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()

# Plot sample car images
car_indices = np.where(y_train.flatten() == 1)[0][:5]  # Car class is labeled as 1
car_images = x_train[car_indices]
plot_images(car_images, ['Car'] * len(car_images))

# Preprocess CIFAR-10 car images
x_train_car = np.array([preprocess_image(image) for image in car_images])

# Load and plot gun images
gun_images = load_images('guns', 4)
plot_images(gun_images, ['Gun'] * len(gun_images))

# Concatenate car and gun images and labels
x_train = np.concatenate((x_train_car, gun_images), axis=0)
y_train = np.concatenate((np.ones(len(x_train_car)), np.zeros(len(gun_images))), axis=0)

# Model Architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Model Compilation
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Model Training
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Save the model
model.save('car_gun_classifier.h5')

# Load test dataset
(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Preprocess test dataset
x_test_car = x_test[y_test.flatten() == 1]
x_test_car = np.array([preprocess_image(image) for image in x_test_car])

gun_images_test = load_images('guns', 4)

# Concatenate car and gun images and labels for test dataset
x_test = np.concatenate((x_test_car, gun_images_test), axis=0)
y_test = np.concatenate((np.ones(len(x_test_car)), np.zeros(len(gun_images_test))), axis=0)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
