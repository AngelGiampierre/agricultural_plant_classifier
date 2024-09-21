import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

### Training (CNN)

# Path
data_dir = 'C:\\Users\\USER\\Downloads\\archive\\Agricultural-crops'

# Parameters
batch_size = 32
image_size = (299, 299)
num_classes = 30
validation_split = 0.2

# Training and validation fata generator
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=validation_split
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Pretrained model InceptionV3
base_model = InceptionV3(weights='imagenet', include_top=False)

# Custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

# Compiling model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training model
epochs = 25
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs
)

# Saving model
model.save('classifier1.keras')


### Test

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

model = tf.keras.models.load_model('classifier1.keras')

test_generator = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Prediction
y_pred = model.predict(test_generator)
y_true = test_generator.classes

class_labels = list(test_generator.class_indices.keys())

# Confusion Matrix
confusion = confusion_matrix(y_true, np.argmax(y_pred, axis=1))
print(classification_report(y_true, np.argmax(y_pred, axis=1), target_names=class_labels))
print("Matriz de confusión:")
print(confusion)