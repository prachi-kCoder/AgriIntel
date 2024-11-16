
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from info import disease_info
import os

app = Flask(__name__)

# Load the model once when the app starts
model = load_model('../MileStone3/PlantAI_CNN_model.keras')

# Configuration for file uploads
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Helper function to check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    image = load_img(image_path, target_size=(224, 224))  # Adjust target size to your model's input size
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize if your model requires
    return image

# Prediction function
def predict_disease(image_path):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction)  # Get the index of the highest probability
    confidence = np.max(prediction) * 100  # Get the confidence level as a percentage
    # Customize this with your actual class names
    class_names = [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
        'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
        'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
        'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
] 
    disease_name = class_names[predicted_class] if predicted_class < len(class_names) else "Unknown"
    return disease_name, confidence

# Home page route
@app.route('/')
def home():
    return render_template('home.html')  # Home page template

# Disease Detector page route
@app.route('/disease-detector')
def disease_detector():
    return render_template('upload.html')  # Upload form template

# File upload and prediction route
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Make prediction using the model
        disease_name, confidence = predict_disease(file_path)

        # Redirect to the result page with prediction data
        return redirect(url_for('result', filename=filename, prediction=disease_name, confidence=confidence))

    return redirect(request.url)

# Result page route to show prediction result
@app.route('/result')
def result():
    filename = request.args.get('filename')
    prediction = request.args.get('prediction')
    confidence = request.args.get('confidence')
    return render_template('result.html', filename=filename, prediction=prediction, confidence=confidence)

# Cure page route based on predicted disease
@app.route('/cure')
def cure():
    disease = request.args.get('disease')
    
    # Default to "Healthy" if disease not found in data
    if disease not in disease_info:
        disease = "Healthy" 
    
    disease_details = disease_info[disease]
    
    # Pass the additional fields to the template
    return render_template(
        'cure.html', 
        disease=disease, 
        description=disease_details.get('description'), 
        cause=disease_details.get('cause'), 
        symptoms=disease_details.get('symptoms', []), 
        prevention=disease_details.get('prevention', []), 
        treatment=disease_details.get('treatment', [])
    )

# About and Contact page routes
@app.route('/information')
def information():
    return render_template('information.html')  # About Us page template

@app.route('/contact')
def contact():
    return render_template('contact.html')  # Contact page template


@app.route('/achievement')
def achievement():
    # Fetch disease information to display on this page (if needed)
    return render_template('achievement.html')


if __name__ == "__main__":
    app.secret_key = 'your_secret_key'
    app.run(debug=True)
    
    
# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']
        
#         # Authenticate the user (placeholder for actual authentication logic)
#         if username == 'admin' and password == 'password':  # Example credentials
#             flash('Login successful!', 'success')
#             return redirect(url_for('home'))
#         else:
#             flash('Invalid credentials, please try again.', 'error')
    
#     return render_template('login.html')

# @app.route('/signup')
# def signup():
#     return render_template('signup.html')
