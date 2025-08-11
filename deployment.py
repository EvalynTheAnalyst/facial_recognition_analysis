from PIL import Image
import tensorflow as tf
import gradio as gr
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array
import os

MODEL_PATH = os.path.join("model","/content/model.h5")
CLASS_NAME = ['angry', 'sad', 'happy','disgust', 'surprise', 'neutral']

model = tf.keras.models.load_model(MODEL_PATH)


def predict_emotion(image:Image,face_detect=True):
  if image is None:
    return {"label": "No image", "confidence": 0.0}

    # Convert to RGB, resize, normalize, and reshape for model
  img = image.convert("RGB").resize((224, 224))
  img_array = img_to_array(img) / 255.0
  img_array = np.expand_dims(img_array, axis=0)

  preds = model.predict(img_array)
  idx = int(np.argmax(preds, axis = 1)[0])
  label = CLASS_NAME[idx]
  confidence = float(np.max(preds))

  return f"Label: {label} \n \n Confidence: {round(confidence,4)}"


  #Creating the theme
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="rose",
    neutral_hue="gray",
    font=["Helvetica", "ui-sans-serif", "system-ui"],
)

  #user interface
demo = gr.Interface(
      fn = predict_emotion,
      inputs = [
          gr.Image(type = 'pil',label = 'Upload a face Image', image_mode='RGB'),
          gr.Checkbox(label = 'Run face and crop', value=True)
      ],
      outputs=[
          gr.Label(num_top_classes=1,label ="Predicted emotions and Confidence")

      ],
      title='Emotion Detector',
      description=(
          "<div style='text-align:center'>"
        "<h3>Upload a photo and let the AI detect the emotion!</h3>"
        "<p>Emotions: angry, disgust, fear, happy, neutral, sad, surprise.</p>"
        "</div>"
      ),
      theme = theme,
      allow_flagging="never",
      live = True,

  )

if __name__ == "__main__":
    demo.launch(debug=True)

