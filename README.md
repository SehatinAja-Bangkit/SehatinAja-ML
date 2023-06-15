# SehatinAja-ML
We Are Deploying the model into 2 type TF lite and TF Serving

## TF Serving
The model is saved first into SavedModel then creates a Tensorflow Function by getting the value from the concrete function first then the frozen model is saved and converted to .pb format so that it can be integrated with Mobile Development and Cloud Computing later
```
import tensorflow as tf
import shutil
from google.colab import drive

drive.mount('/content/gdrive')

# Save the trained model as SavedModel
model.save("/content/gdrive/MyDrive/Coursera/model")

# Load the model and compile it
loaded_model = tf.keras.models.load_model("/content/gdrive/MyDrive/Coursera/model")
loaded_model.compile(optimizer=tf.keras.optimizers.Adam(), loss="binary_crossentropy", metrics=["accuracy"])

# Convert SavedModel to .pb format
concrete_function = loaded_model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

# Create a tf.function to get the concrete function
@tf.function(input_signature=[tf.TensorSpec(shape=(None, 64, 64, 3), dtype=tf.float32, name="image_bytes")])
def serving_fn(input):
    return concrete_function(input)

# Get the concrete function
concrete_function = serving_fn.get_concrete_function()

# Save the frozen graph
pb_path = "/content/gdrive/MyDrive/Coursera/model/model.pb"
tf.io.write_graph(graph_or_graph_def=concrete_function.graph,
                  logdir='.',
                  name=pb_path,
                  as_text=False)

print("Model saved in .pb format.")

# Download the .pb file
destination_path = "/content/gdrive/MyDrive/Coursera/model.pb"
shutil.move(pb_path, destination_path)
print("Model downloaded.")


 ```
 
 
 ## TF Lite
 For TFLite first we need to convert the SavedModel into quantization and after that we Save the model once again and download the .tflite format 
 ```
 from google.colab import files

# Convert the model to TFLite with quantization configuration
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

# Save the TFLite model to a file
tflite_file_path = '/content/quantized_model.tflite'
with open(tflite_file_path, 'wb') as f:
    f.write(tflite_model)

# Download the TFLite file
files.download(tflite_file_path)
 ```
 
 ## Result
### Accuracy
Almost 100% after 100 Epochs

![image](https://github.com/SehatinAja-Bangkit/SehatinAja-ML/assets/92217354/093ebdb5-6f20-439c-b80a-9f9fdbf109ab)

### Loss
Almost 0 Loss After 100 Epochs

![image](https://github.com/SehatinAja-Bangkit/SehatinAja-ML/assets/92217354/78f74324-c743-454f-87aa-8f424f566a84)
