from flask import Flask, jsonify, request
import subprocess

app = Flask(__name__)

@app.route("/")
def landing():
  return "Post to /evaluate-model"


@app.route('/evaluate-model', methods=['POST'])
def evaluate_model():
    text = request.get_data()
    print(text)

    


    result = subprocess.run(
        ['python', '../hw4/code/assignment.py', 'LOAD'],
        capture_output=True, shell=True)

    # Get the output as a string
    # output = result.stdout.decode('utf-8')


    print(result.stdout)
    print('\n')
    print(result.stderr)

    return '', 204