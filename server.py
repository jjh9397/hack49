from flask import Flask, request, jsonify
from flask_cors import CORS
from ml.classify import evaluateFile
import sys
import os

app = Flask(__name__)

CORS(app)

app.config['UPLOAD_FOLDER'] = 'userfiles'
# app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
    

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)


    processed_result, val = evaluateFile(file_path)

    # Return the result as JSON
    return jsonify({"filename": file.filename, "prediction": processed_result, "prediction_val": val})

if __name__ == '__main__':
    app.run(debug=True, port=5000)

    