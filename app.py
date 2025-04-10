from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Allows all origins (for testing)

# Set upload folder and allowed extensions for file uploads
UploadFolder = '/path/to/upload/folder'  # Modify with the actual path
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'csv'}  # Add more file types if needed

# Configure Flask app
app.config['UploadFolder'] = UploadFolder
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max file size: 16 MB

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_text(filename):
    file_path = os.path.join(app.config['UploadFolder'], filename)
    
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
            file.save(os.path.join(app.config['UploadFolder'], filename))
            
            # Process the uploaded text or CSV file
            file_content = process_text(filename)
            
            return jsonify({'message': 'File uploaded successfully', 'content': file_content}), 200
        else:
            return jsonify({'error': 'File type not allowed'}), 400

    # If no file is uploaded, proceed with processing text (when sending JSON data)
    output_text = process_text(input_text)
    
    return output_text

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
