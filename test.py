import os
from PIL import Image
import tensorflow as tf
import numpy as np
from tkinter import filedialog
from tkinter import Tk

# Trained model
model = tf.keras.models.load_model('classifier1.keras')

def resize_image(image_path):
    image = Image.open(image_path)
    return image.resize((224, 224))

def classify_image(image, threshold=0.5):
    image = np.asarray(image) / 255.0
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)

    class_label = np.argmax(predictions)
    confidence = np.max(predictions)
    
    # Metrics
    top_classes = np.argsort(predictions)[0, ::-1][:3]
    top_confidences = predictions[0, top_classes]
    class_difference = np.diff(top_confidences).mean()
    uncertainty = 1 - confidence
    entropy = -np.sum(predictions * np.log2(predictions + 1e-10))
    
    multiple_classes = class_difference > threshold
    
    return class_label, confidence, uncertainty, entropy, multiple_classes


# Select image from PC
def classify():
    root = Tk()
    root.withdraw()

    # Select file
    file_path = filedialog.askopenfilename(title="Seleccionar archivo PNG", filetypes=[("Archivos PNG", "*.png")])

    if not file_path:
        print("No se seleccionó ningún archivo.")
        exit()
        
    if file_path:
        resized_image = resize_image(file_path)

        # Classifying
        class_label, confidence, uncertainty, entropy, multiple_classes = classify_image(resized_image)

        # Classes
        class_labels = {
            0: "Cherry",
            1: "Coffe-plant",
            2: "Cucumber",
            3: "Fox_nut(Makhana)",
            4: "Lemon",
            5: "Olive-tree",
            6: "Pearl_millet(bajra)",
            7: "Tobacco-plant",
            8: "almond",
            9: "banana",
            10: "cardamom",
            11: "chilli",
            12: "clove",
            13: "coconut",
            14: "cotton",
            15: "gram",
            16: "jowar",
            17: "jute",
            18: "maize",
            19: "mustard-oil",
            20: "papaya",
            21: "pineapple",
            22: "rice",
            23: "soyabean",
            24: "sugarcane",
            25: "sunflower",
            26: "tea",
            27: "tomato",
            28: "vigna-radiati(Mung)",
            29: "wheat",
        }

        # Results
        result_label = f"Agricultural plant: {class_labels[class_label]}"
        print(result_label)
        print('confidence: ', confidence)
        print('uncertainty: ', uncertainty)
        print('entropy: ', entropy)
        print('multiple: ', multiple_classes)

if __name__ == "__main__":
    classify()