from flask import Flask, jsonify, request
from subprocess import Popen, PIPE, STDOUT
from flask_cors import CORS
from simplify import simplify_main
from evaluate import evaluate_main

app = Flask(__name__)
CORS(app)

@app.route("/")
def landing():
  return "GET at /evaluate or POST to /simplify"


# @app.route('/evaluate-model', methods=['POST', 'GET'])
# def evaluate_model():
#   if request.method == 'POST':
#     text = request.get_data()
#     print(text)
    
#     p = Popen('python ../code/main.py LOAD', stdout = PIPE, 
#         stderr = PIPE, shell = True)
#     # p = Popen('python ../code/help.py', stdout = PIPE, 
#     #       stderr = PIPE, shell = True)
    
#     stdout = ""
#     stderr = ""
    
#     while True:
#       line_out = p.stdout.readline()
#       line_err = p.stderr.readline()
#       if (not line_out and not line_err): break
#       elif not line_out:
#         # print(line_err)
#         stderr += line_err.decode("utf-8")
#       elif not line_err:
#         # print(line_out)
#         stdout += line_out.decode("utf-8")
#       else:
#         # print(line_out)
#         # print(line_err)
#         stdout += line_out.decode("utf-8")
#         stderr += line_err.decode("utf-8")

#     # print(stdout)
#     # print(stderr)
#     return jsonify({
#             "stdout": stdout,
#             "stderr" : stderr,
#             "METHOD" : "POST"
#         })
#   else:
#     return "Send a POST request to this URL with a text string of the sentence(s) you want to simplify"

@app.route('/evaluate', methods=['GET'])
def evaluate_model():
  if request.method == 'GET':
    
    loss, acc, perp = evaluate_main()

    # print(stdout)
    # print(stderr)
    return jsonify({
            "loss": loss,
            "accuracy": acc,
            "perplexity": perp,
        })
  
@app.route('/simplify', methods=['POST', 'GET'])
def simplify():
  if request.method == 'POST':
    text = request.get_data().decode("utf-8")
    print(text)
    
    simplified_text = simplify_main(text)
    
    return jsonify({
            "text": simplified_text,
            "METHOD" : "POST"
        })
  else:
    return "Send a POST request to this URL with a text string of the sentence(s) you want to simplify"
    
if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)