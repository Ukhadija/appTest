from flask import Flask, request, jsonify

app = Flask(__name__)

def TagGeneration(input_text):
    # Your custom processing logic here
    return input_text.upper()  # Example: Convert input to uppercase

@app.route('/tagGeneration', methods=['POST'])
def process():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'Missing text field'}), 400
    
    input_text = data['text']
    output_text = TagGeneration(input_text)
    
    return jsonify({'input': input_text, 'output': output_text})

if __name__ == '__main__':
    app.run(debug=True)
