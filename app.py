from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allows all origins (for testing)

def process_text(input_text):
    return input_text.upper()

@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'Missing text field'}), 400
    
    input_text = data['text']
    output_text = process_text(input_text)
    
    return jsonify({'input': input_text, 'output': output_text})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
