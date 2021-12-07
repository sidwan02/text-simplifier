from flask import Flask, jsonify, request
import subprocess

app = Flask(__name__)

@app.route("/")
def landing():
  return "Post to /evaluate-model"


@app.route('/evaluate-model', methods=['POST', 'GET'])
def evaluate_model():
  if request.method == 'POST':
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

    return result.stdout, 204
  else:
    return "Send a POST request to this URL with a text string of the sentence(s) you want to simplify"
    
if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)