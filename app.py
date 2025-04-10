import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Allows all origins (for testing)

# Set the base directory and upload folder relative to app.py
BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # Get the directory where app.py is located
UploadFolder = os.path.join(BASE_DIR, 'uploads')  # Set upload folder relative to app.py

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'csv'}  # Add more file types if needed

# Ensure the upload folder exists
if not os.path.exists(UploadFolder):
    os.makedirs(UploadFolder)

# Configure Flask app
app.config['UPLOAD_FOLDER'] = UploadFolder
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max file size: 16 MB

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_text(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Check if file exists
    if not os.path.exists(file_path):
        return None
    
    # Open the file and read the content
    with open(file_path, 'r') as file:
        file_content = file.read()
    
    os.remove(file_path)
    return file_content

@app.route('/process', methods=['POST'])
def process():
    # If the request contains a file
    if 'file' in request.files:
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            # Secure the filename to prevent directory traversal attacks
            filename = secure_filename(file.filename)
            
            # Save the file to the upload folder
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            # Process the uploaded text or CSV file
            file_content = process_text(filename)
            
            if not file_content:
                return jsonify({'error': 'Unable to process the file'}), 500
            
            return jsonify({'message': 'File uploaded successfully', 'content': file_content}), 200
        else:
            return jsonify({'error': 'File type not allowed'}), 400

    return jsonify({'error': 'No file part'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
