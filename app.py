import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input


# Function to load model and perform Grad-CAM
def grad_cam(model, img):
    # BGR2RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize image
    img = cv2.resize(img, (64, 64))
    # Expand dims
    img = np.expand_dims(img, axis=0)

    # Generate prediction
    pred = model.predict(img)

    # Generate prediction class
    pred_output = (pred > 0.49).astype(int)[0][0]

    # Feature map of 'f1' layer, which is the last convolution layer
    last_conv_layer = model.get_layer('f1')
    # Create functional model
    last_conv_layer_model = Model(model.inputs, last_conv_layer.output)

    # New model's input shape
    classifier_input = Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input

    # My model's classification layers (add layer names from last conv layer up to prediction layer)
    classifier_layer_names = ['max_pooling2d_23', 'batch_normalization_29', 'dropout_13', 'global_average_pooling2d',
                              'dense_12', 'batch_normalization_30', 'dropout_14']

    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)

    # Functional model creation
    classifier_model = Model(classifier_input, x)

    # Compute gradients
    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(img)
        tape.watch(last_conv_layer_output)

        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # Calculate heat map
    heatmap = np.mean(last_conv_layer_output, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap = cv2.resize(heatmap, (64, 64))
    heatmap = heatmap.reshape((64, 64))

    return heatmap, pred_output


# Load the model
model = load_model("model_64_64_3.h5")
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://th.bing.com/th/id/R.759e79988958d8231e20b326023bf54a?rik=6ZxV4RTmc3MqwQ&riu=http%3a%2f%2fgetwallpapers.com%2fwallpaper%2ffull%2ff%2f8%2f0%2f1353989-beautiful-web-wallpaper-1920x1080.jpg&ehk=IAecvhArMaMEM8NGfsTdY7oNIch9wZ8jsh5XAECovsA%3d&risl=&pid=ImgRaw&r=0");
background-size: 100%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
st.title("Malaria Detection using Deep Learning")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded file
    img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Generate Grad-CAM heatmap and get predicted class
    heatmap, pred_class = grad_cam(model, img)

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the original image on the left subplot
    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    # Plot the overlayed image on the right subplot
    axs[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[1].imshow(heatmap, cmap='jet', alpha=0.75)  # Overlay heatmap with reduced opacity
    axs[1].set_title('Overlayed Image')
    axs[1].axis('off')

    # Print predicted class
    if(pred_class):
        st.write("The given cell is Parasitized !!")
    else:
        st.write("The given cell is Uninfected")

    # Display the plot
    st.pyplot(fig)


