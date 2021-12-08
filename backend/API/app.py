from flask import Flask, jsonify, request
import subprocess
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

@app.route("/")
def landing():
  # result = subprocess.run(
  #       ['python', '../hw4/code/assignment.py', 'LOAD'],
  #       capture_output=True, shell=True)

  # # Get the output as a string
  # # output = result.stdout.decode('utf-8')


  # print(result.stdout)
  # print('\n')
  # print(result.stderr)
  
  from subprocess import Popen, PIPE, STDOUT

  # p = Popen('python ../hw4/code/assignment.py LOAD', stdout = PIPE, 
  #       stderr = STDOUT, shell = True)
  p = Popen('python ../hw4/code/help.py', stdout = PIPE, 
        stderr = PIPE, shell = True)
  
  while True:
    line_out = p.stdout.readline()
    line_err = p.stderr.readline()
    if (not line_out and not line_err): break
    elif not line_out:
      print(line_err)
    elif not line_err:
      print(line_out)
    else:
      print(line_out)
      print(line_err)
  
  return "Post to /evaluate-model"


@app.route('/evaluate-model', methods=['POST', 'GET'])
def evaluate_model():
  if request.method == 'POST':
    text = request.get_data()
    print(text)
    
    # status, result = subprocess.getstatusoutput("python test.py")
    # print("status: ", status)
    # print("result: ", result)
    

    result = subprocess.run(
        ['python', '../hw4/code/assignment.py', 'LOAD'],
        capture_output=True, shell=True)

    # Get the output as a string
    # output = result.stdout.decode('utf-8')


    print(result.stdout)
    print('\n')
    print(result.stderr)

    return result.stdout, 204
    # return "booboo"
  else:
    return "Send a POST request to this URL with a text string of the sentence(s) you want to simplify"
    
if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)