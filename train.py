import tensorflow as tf
from tensorflow.keras import layers, models

img_size = 224
batch_size = 16

dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    image_size=(img_size, img_size),
    batch_size=batch_size
)

class_names = dataset.class_names
print("Classes:", class_names)

dataset = dataset.map(lambda x, y: (x/255.0, y))

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(dataset, epochs=25)

model.save("wear_model.h5")

# save labels
with open("labels.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")