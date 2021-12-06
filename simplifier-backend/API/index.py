from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/metric-values', methods=['POST'])
def get_metric_scores_from_text():
    text = request.get_data()
    print(text)
    return '', 204

@app.route('/simplified-text', methods=['POST'])
def get_simplified_text_from_text():
    text = request.get_data()
    print(text)
    return '', 204