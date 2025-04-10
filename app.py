from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import os
import re
import random
import itertools
import csv
from langchain_community.llms import HuggingFacePipeline
from langchain import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from qdrant_client import QdrantClient
from sklearn.metrics.pairwise import cosine_similarity
from qdrant_client.models import Distance, VectorParams
import nomic
from nomic import embed
from qdrant_client.http import models
from qdrant_client.models import PointStruct
import numpy as np
import csv
from qdrant_client.http import models
from io import StringIO
from pathlib import Path

counter = 0


app = Flask(__name__)
CORS(app)  # Allows all origins (for testing)

# Set upload folder and allowed extensions for file uploads
UploadFolder = 'UploadFolder'  # Modify with the actual path
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



@app.route('/process', methods=['POST'])
def process():
    # If the request contains a file
    try:
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
                file_content = filename + "ok"
                
                if not file_content:
                    return jsonify({'error': 'Unable to process the file'}), 500
                
                return jsonify({'message': 'File uploaded successfully', 'content': file_content}), 200
            else:
                return jsonify({'error': 'File type not allowed'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
