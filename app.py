import pandas as pd
import os
import numpy as np
import cv2 # Import OpenCV for image processing
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.models import load_model, Model # type: ignore # Import Model
from tensorflow.keras.applications import DenseNet121 # type: ignore # Import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout # type: ignore
import tensorflow.keras.backend as K # type: ignore
import tensorflow as tf # Import TensorFlow
import google.generativeai as genai
from datetime import datetime

app = Flask(_name_)

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'static/uploads'
MODIFIED_FOLDER = 'static/modified'
EXAMPLE_IMAGES_FOLDER = 'static/dr_examples'
HEATMAP_FOLDER = 'static/heatmaps' # New folder for Grad-CAM heatmaps
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['EXAMPLE_IMAGES_FOLDER'] = EXAMPLE_IMAGES_FOLDER
app.config['HEATMAP_FOLDER'] = HEATMAP_FOLDER # Add heatmap folder to config

IMG_WIDTH = 224
IMG_HEIGHT = 224

diagnosis_map = {
    0: 'No_DR',
    1: 'Mild',
    2: 'Moderate',
    3: 'Severe',
    4: 'Proliferate_DR'
}
reverse_diagnosis_map = {v: k for k, v in diagnosis_map.items()}
class_names = list(diagnosis_map.values())

dr_to_text_map = {
    'No_DR': 'No Diabetics<br>Cardiac Risk Absent<br>Glaucoma Risk Absent<br>Healthy Retina Observed',
    'Mild': 'Mild Diabetics<br>Low Cardiac Risk<br>Low Glaucoma Risk<br>Few Microaneurysms Seen.',
    'Moderate': 'Moderate Diabetics<br>Cardiac Risk Seen.<br>Glaucoma Risk Seen.<br>More lesions present.',
    'Severe': 'Severe Diabetics.<br>High Cardiac Risk.<br>Presence of Glaucoma.<br>Extensive Damage Noted.',
    'Proliferate_DR': 'Proliferate Diabetics<br>High Cardiac Risk.<br>High Glaucoma Risk.<br>Neovascualrization Present.'
}

dr_example_images = {
    'No_DR': 'no_dr.png',
    'Mild': 'mild_dr.png',
    'Moderate': 'moderate_dr.png',
    'Severe': 'severe_dr.png',
    'Proliferate_DR': 'proliferate_dr.png'
}

# --- END CONFIGURATION ---


# --- Model Loading ---
MODEL_WEIGHTS_PATH = 'diabetic_retinopathy_classification_weights.weights.h5'
loaded_model = None
grad_model = None # Model for Grad-CAM

# *** CORRECTED LAST CONVOLUTIONAL LAYER NAME - Ensure this line is exactly as shown ***
last_conv_layer_name = 'conv3_block12_concat'


def build_gradcam_model(model, last_conv_layer_name):
    """Builds a model that maps the input image to the activations of the last conv layer and the final predictions."""
    if model is None:
        print("Cannot build Grad-CAM model: Loaded model is None.")
        return None
    try:
        # Find the layer by name
        last_conv_layer = model.get_layer(last_conv_layer_name)
        # Create a model that outputs the last convolutional layer's activations and the model's final predictions
        grad_model = Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]
        )
        print(f"Successfully built Grad-CAM model using layer: {last_conv_layer_name}")
        return grad_model
    except ValueError as e:
        print(f"Error building Grad-CAM model: Layer '{last_conv_layer_name}' not found in the model.")
        print("Please check the layer name in your DenseNet121 model architecture.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while building Grad-CAM model: {e}")
        return None


def load_trained_model(weights_path):
    """Loads the pre-trained DenseNet121 model and its weights."""
    try:
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        num_classes = len(diagnosis_map)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        absolute_weights_path = os.path.join(app.root_path, weights_path)

        if os.path.exists(absolute_weights_path):
            model.load_weights(absolute_weights_path)
            print(f"Successfully loaded model weights from {absolute_weights_path}")
        else:
            print(f"Warning: Model weights file not found at '{absolute_weights_path}'.")
            print("Prediction and Grad-CAM will not work correctly until weights are trained and saved.")

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    except Exception as e:
        print(f"Error loading model or weights: {e}")
        print("Please ensure TensorFlow and other libraries are installed correctly.")
        return None

# Attempt to load the model and build the Grad-CAM model on startup
loaded_model = load_trained_model(MODEL_WEIGHTS_PATH)
if loaded_model:
    grad_model = build_gradcam_model(loaded_model, last_conv_layer_name)


# --- End Model Loading ---


# --- Gemini Chatbot Configuration ---
GEMINI_API_KEY = "AIzaSyCOR-Y14uiPPNWapMyelZ9oO4xmP55YWdU"

gemini_model = None
chat_session = None

if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY":
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Using the model name that you found works
        gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        print("Gemini API configured.")

    except Exception as e:
        print(f"Error configuring Gemini API. Chatbot and summary generation will be unavailable: {e}")
        gemini_model = None # Ensure model is None if configuration fails
else:
    print("Warning: GEMINI_API_KEY not set or is placeholder. Chatbot and summary generation will be unavailable.")

# --- End Gemini Chatbot Configuration ---


# --- Grad-CAM Helper Functions ---

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Computes the Grad-CAM heatmap for a given image array."""
    if model is None or grad_model is None:
        print("Grad-CAM model not available.")
        return None

    try:
        # Use the Grad-CAM model to get the activations and predictions
        with tf.GradientTape() as tape: # Use TensorFlow's GradientTape
            last_conv_layer_output, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0]) # Get the predicted class index

            # Get the loss for the predicted class
            class_channel = predictions[:, pred_index]

        # Compute the gradient of the top predicted class with respect to the output feature map of the last convolutional layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # Get the mean intensity of the gradient over each feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Multiply each channel in the feature map array by "how important" this channel is with regard to the top predicted class
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis] # Use tf.newaxis for broadcasting
        heatmap = tf.squeeze(heatmap) # Remove single-dimensional entries

        # Normalize the heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
        return heatmap.numpy() # Return as numpy array

    except Exception as e:
        print(f"Error generating Grad-CAM heatmap: {e}")
        # This might happen if gradients are None, which can occur with certain models or inputs
        return None


def save_and_display_gradcam(img_path, heatmap, output_path="static/heatmaps/heatmap.png"):
    """Overlays the heatmap on the original image and saves the result."""
    if heatmap is None:
        print("Cannot save Grad-CAM: Heatmap is None.")
        return None

    try:
        # Load the original image using OpenCV
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading image for Grad-CAM overlay: {img_path}")
            return None

        # Resize the heatmap to match the original image size
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        # Convert heatmap to BGR for visualization
        heatmap = np.uint8(255 * heatmap)
        # Apply a colormap (e.g., COLORMAP_JET)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Superimpose the heatmap on the original image
        # Adjust alpha (0.4 here) for transparency
        superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

        # Ensure the heatmap output folder exists
        heatmap_folder_absolute = os.path.join(app.root_path, HEATMAP_FOLDER)
        if not os.path.exists(heatmap_folder_absolute):
             os.makedirs(heatmap_folder_absolute)

        # Generate a unique filename for the heatmap overlay
        original_filename = os.path.basename(img_path)
        filename_base, file_extension = os.path.splitext(original_filename)
        heatmap_filename = f"{filename_base}_heatmap.png" # Save as PNG
        output_path_absolute = os.path.join(heatmap_folder_absolute, heatmap_filename)

        # Save the superimposed image
        cv2.imwrite(output_path_absolute, superimposed_img)

        # Return the URL relative to the static folder
        path_relative_to_static = os.path.join('heatmaps', heatmap_filename).replace(os.sep, '/')
        return url_for('static', filename=path_relative_to_static)

    except Exception as e:
        print(f"Error saving or displaying Grad-CAM: {e}")
        return None

# --- End Grad-CAM Helper Functions ---


# --- Helper Functions ---
def allowed_file(filename):
    """Checks if the uploaded filename has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    """
    Loads and preprocesses the image, makes a prediction using the loaded model,
    and returns the comprehensive text prediction and the original image ID.
    """
    # Removed raw_predictions from the return value
    if loaded_model is None:
        return "Model weights not loaded. Cannot make prediction.", None, None, None

    try:
        img = image.load_img(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        predictions = loaded_model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_names[predicted_class_index]

        comprehensive_prediction = dr_to_text_map.get(predicted_class_name, f"Prediction: {predicted_class_name}")

        image_id = os.path.splitext(os.path.basename(image_path))[0]

        # Removed raw_predictions from the return value
        return comprehensive_prediction, image_id, predicted_class_name, predicted_class_index

    except Exception as e:
        print(f"Error during prediction: {e}")
        # Removed None for raw_predictions
        return f"Error during prediction: {e}", None, None, None

def find_matching_modified_image(image_id):
    """Searches for a matching image (with .png extension) in the static/modified folder."""
    target_filename = image_id + ".png"
    absolute_modified_image_path = os.path.join(app.root_path, MODIFIED_FOLDER, target_filename)

    if os.path.exists(absolute_modified_image_path):
        path_relative_to_static = os.path.join('modified', target_filename).replace(os.sep, '/')
        return path_relative_to_static

    return None

def get_example_image_url(predicted_class_name):
    """
    Gets the URL for the example image corresponding to the predicted class name.
    Returns the URL or None if not found.
    """
    example_filename = dr_example_images.get(predicted_class_name)
    if example_filename:
        path_relative_to_static = os.path.join('dr_examples', example_filename).replace(os.sep, '/')
        absolute_example_image_path = os.path.join(app.root_path, EXAMPLE_IMAGES_FOLDER, example_filename)
        if os.path.exists(absolute_example_image_path):
            return url_for('static', filename=path_relative_to_static)
        else:
            print(f"Warning: Example image file not found at '{absolute_example_image_path}' for class '{predicted_class_name}'.")
            return None
    return None

def generate_result_summary(predicted_class_name):
    """
    Uses Gemini to generate a short explanation and relevant information based on the predicted DR class.
    Returns a dictionary with 'explanation' and 'precautions' or None on error.
    """
    if gemini_model is None:
        print("Model not available for summary generation.")
        return None

    # Modified prompt to remove "AI" and focus on "Assistant" or "analysis"
    prompt = f"""
    As a medical assistant for a Diabetic Retinopathy analysis tool, please provide:
    1. A brief explanation of the predicted diagnosis "{predicted_class_name}".
    2. Relevant general information or precautions related to this diagnosis and its potential links to cardiac and glaucoma risks, based on common medical understanding (but not medical advice).

    Format your response strictly as follows:
    EXPLANATION: [Your concise explanation here]
    PRECAUTIONS: [Your relevant information/precautions here]

    Keep both the explanation and precautions concise (1-3 sentences each).
    Reiterate that this tool is for informational purposes and not medical advice.
    """

    print(f"Sending prompt to Assistant for summary: {prompt}") # Changed log message

    try:
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()

        print(f"Received raw response from Assistant for summary:\n{response_text}") # Changed log message

        explanation = "Could not generate explanation."
        precautions = "Could not generate precautions."

        if "EXPLANATION:" in response_text and "PRECAUTIONS:" in response_text:
            try:
                parts = response_text.split("PRECAUTIONS:", 1)
                explanation_part = parts[0].replace("EXPLANATION:", "", 1).strip()
                precautions_part = parts[1].strip()

                explanation = explanation_part if explanation_part else explanation
                precautions = precautions_part if precautions_part else precautions

            except Exception as parse_error:
                 print(f"Error parsing Assistant summary response: {parse_error}") # Changed log message
                 print(f"Raw response that failed parsing:\n{response_text}")
                 explanation = f"Summary parsing error. Raw response snippet: {response_text[:100]}..."
                 precautions = "Please consult a healthcare professional."


        else:
            print(f"Warning: Assistant response for summary did not contain expected markers (EXPLANATION:, PRECAUTIONS:):\n{response_text}") # Changed log message
            explanation = f"Summary format error. Raw response: {response_text[:100]}..."
            precautions = "Please consult a healthcare professional."


        return {"explanation": explanation, "precautions": precautions}

    except Exception as e:
        print(f"Error generating result summary with Assistant: {e}") # Changed log message
        return {"explanation": f"Summary generation failed: {e}", "precautions": "Please consult a healthcare professional."}

def generate_xai_explanation(predicted_class_name):
    """
    Uses Gemini to generate an explanation about what features might be highlighted
    by the model for the given predicted class, in the context of a heatmap.
    """
    if gemini_model is None:
        print("Model not available for focus area explanation.") # Changed log message
        return "Focus areas could not be determined." # Changed text

    # Modified prompt to focus on heatmap visualization
    prompt = f"""
    As a medical assistant for a Diabetic Retinopathy analysis tool, explain what visual features in a retinal image the heatmap is highlighting for the prediction "{predicted_class_name}".

    Focus on describing the types of lesions or features typically associated with this stage that would be highlighted (e.g., microaneurysms, hemorrhages, exudates, neovascularization). Explain that the heatmap indicates areas the model focused on.

    Keep the explanation concise (1-2 sentences).
    """

    print(f"Sending prompt to Assistant for heatmap explanation: {prompt}") # Changed log message

    try:
        response = gemini_model.generate_content(prompt)
        xai_explanation = response.text.strip()
        print(f"Received raw response from Assistant for heatmap explanation:\n{xai_explanation}") # Changed log message
        return xai_explanation if xai_explanation else "Could not generate explanation for heatmap." # Changed text

    except Exception as e:
        print(f"Error generating heatmap explanation with Assistant: {e}") # Changed log message
        return f"Error generating explanation for heatmap: {e}" # Changed text


# --- End Helper Functions ---


# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    """Renders the main index page."""
    return render_template('index.html', now=datetime.now)


@app.route('/about')
def about():
    """Renders the About page."""
    return render_template('about.html', now=datetime.now)


@app.route('/predict', methods=['POST'])
def predict():
    """Handles image upload, prediction, and returns results as JSON."""

    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
         return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        uploaded_image_path_relative_for_save = os.path.join(UPLOAD_FOLDER, filename)
        uploaded_image_path_absolute = os.path.join(app.root_path, uploaded_image_path_relative_for_save)

        uploads_folder_absolute = os.path.join(app.root_path, UPLOAD_FOLDER)
        if not os.path.exists(uploads_folder_absolute):
             os.makedirs(uploads_folder_absolute)


        try:
            file.save(uploaded_image_path_absolute)
        except Exception as e:
             print(f"Error saving uploaded file: {e}")
             return jsonify({"error": f"Error saving uploaded file: {e}"}), 500


        # Perform prediction
        # Removed raw_predictions from the returned values
        prediction_text, image_id, predicted_class_name, predicted_class_index = predict_image(uploaded_image_path_absolute)

        if image_id is None:
            if os.path.exists(uploaded_image_path_absolute):
                try:
                    os.remove(uploaded_image_path_absolute)
                    print(f"Cleaned up failed upload: {uploaded_image_path_absolute}")
                except Exception as e:
                    print(f"Error cleaning up file {uploaded_image_path_absolute}: {e}")
            return jsonify({"error": prediction_text}), 500


        # Find the corresponding modified image path
        matching_modified_image_path_relative = find_matching_modified_image(image_id)

        # Generate summary and precautions using the Assistant (formerly Gemini)
        summary_data = generate_result_summary(predicted_class_name)

        # Get the URL for the example image
        example_image_url = get_example_image_url(predicted_class_name)

        # --- Generate Grad-CAM Heatmap and Focus Area Explanation ---
        heatmap_overlay_url = None
        xai_explanation_text = "Heatmap could not be generated or explained." # Changed text

        if grad_model is not None and predicted_class_index is not None:
             try:
                 # Load the image again specifically for Grad-CAM (as a tensor)
                 img = image.load_img(uploaded_image_path_absolute, target_size=(IMG_WIDTH, IMG_HEIGHT))
                 img_array = image.img_to_array(img)
                 img_array = np.expand_dims(img_array, axis=0)
                 img_array /= 255.0 # Normalize

                 # Generate heatmap
                 heatmap = make_gradcam_heatmap(img_array, loaded_model, last_conv_layer_name, predicted_class_index)

                 if heatmap is not None:
                     # Save and get URL for the heatmap overlay image
                     heatmap_overlay_url = save_and_display_gradcam(uploaded_image_path_absolute, heatmap)

                     # Generate XAI explanation using the Assistant (formerly Gemini)
                     xai_explanation_text = generate_xai_explanation(predicted_class_name)
                 else:
                      xai_explanation_text = "Could not generate heatmap."


             except Exception as e:
                 print(f"Error during Grad-CAM or heatmap explanation generation: {e}") # Changed log message
                 heatmap_overlay_url = None
                 xai_explanation_text = f"Error generating explanation for heatmap: {e}" # Changed text
        else:
             print("Grad-CAM model or predicted class index not available. Skipping heatmap generation.")


        # Prepare response data
        response_data = {
            "prediction": prediction_text,
            "image_id": image_id,
            "predicted_class": predicted_class_name,
            "uploaded_image_url": url_for('static', filename=f'uploads/{filename}'),
            "modified_image_url": url_for('static', filename=matching_modified_image_path_relative) if matching_modified_image_path_relative else None,
            "modified_image_not_found": matching_modified_image_path_relative is None, # Flag for frontend
            "gemini_explanation": summary_data.get("explanation", "Could not generate explanation.") if summary_data else "Could not generate explanation.",
            "gemini_precautions": summary_data.get("precautions", "Could not generate precautions.") if summary_data else "Could not generate precautions.",
            "example_image_url": example_image_url,
            "heatmap_overlay_url": heatmap_overlay_url, # Include heatmap URL
            "xai_explanation": xai_explanation_text # Include XAI explanation text
            # Removed "probabilities": raw_predictions
        }

        return jsonify(response_data)

    return jsonify({"error": "File type not allowed"}), 400


@app.route('/chatbot', methods=['POST'])
def chatbot_response():
    """Handles chat messages and interacts with the Assistant (formerly Gemini) API."""
    global chat_session

    if gemini_model is None:
        return jsonify({"response": "Assistant is not available due to configuration issues."}), 500 # Changed text

    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"response": "No message received."}), 400

    try:
        if chat_session is None:
            # Modified context to remove "AI"
            initial_context = """
            You are a medical assistant specialized in Diabetic Retinopathy (DR) and its associated risks (cardiac, glaucoma) based on retinal image analysis.
            Your purpose is to help users understand the analysis results (like DR severity: No_DR, Mild, Moderate, Severe, Proliferate_DR) and provide general, relevant information about these specific conditions.
            You are NOT a substitute for a doctor. You cannot provide medical diagnosis, treatment plans, or advice for any health issue beyond explaining the analysis results in the context of DR and its mentioned associated risks.
            If the user asks about any topic outside of Diabetic Retinopathy, cardiac risk, or glaucoma risk as they relate to retinal image analysis, politely inform them that you can only discuss topics relevant to the retinal analysis and associated risks and cannot provide general medical advice.
            Keep your responses informative, relevant, and always remind the user to consult a qualified healthcare professional for any medical concerns.
            """
            chat_session = gemini_model.start_chat(history=[
                {"role": "user", "parts": [initial_context]},
                # Modified bot's initial response
                {"role": "model", "parts": ["Understood. I am ready to assist with questions related to Diabetic Retinopathy analysis and its associated cardiac and glaucoma risks, as identified by this tool. Please remember I am not a substitute for a doctor."]}
            ])
            response = chat_session.send_message(user_message)

        else:
            response = chat_session.send_message(user_message)

        bot_response = response.text
        return jsonify({"response": bot_response})

    except Exception as e:
        print(f"Error interacting with Assistant API: {e}") # Changed log message
        return jsonify({"response": f"Sorry, I couldn't process that. An error occurred."}), 500

# Removed the /feedback route entirely


# --- App Startup ---
if _name_ == '_main_':
    import tensorflow as tf # Import TensorFlow here to avoid potential issues before app context

    # Ensure necessary static subdirectories exist when the app starts
    static_folder_absolute = os.path.join(app.root_path, 'static')
    uploads_folder_absolute = os.path.join(app.root_path, UPLOAD_FOLDER)
    modified_folder_absolute = os.path.join(app.root_path, MODIFIED_FOLDER)
    example_images_folder_absolute = os.path.join(app.root_path, EXAMPLE_IMAGES_FOLDER)
    heatmap_folder_absolute = os.path.join(app.root_path, HEATMAP_FOLDER) # Heatmap folder


    if not os.path.exists(static_folder_absolute):
         os.makedirs(static_folder_absolute)
         print(f"Created static folder at {static_folder_absolute}")

    if not os.path.exists(uploads_folder_absolute):
        os.makedirs(uploads_folder_absolute)
        print(f"Created uploads folder at {uploads_folder_absolute}")

    if not os.path.exists(modified_folder_absolute):
         print(f"Warning: Modified images folder not found at {modified_folder_absolute}. Modified images will not be displayed.")

    if not os.path.exists(example_images_folder_absolute):
        os.makedirs(example_images_folder_absolute)
        print(f"Created example images folder at {example_images_folder_absolute}. Please add your example images here.")

    # Create the heatmap folder
    if not os.path.exists(heatmap_folder_absolute):
        os.makedirs(heatmap_folder_absolute)
        print(f"Created heatmap folder at {heatmap_folder_absolute}.")


    app.run(debug=True)