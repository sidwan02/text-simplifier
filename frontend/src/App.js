import './App.css';
import { AwesomeButton } from 'react-awesome-button';
import 'react-awesome-button/dist/styles.css';
import { useFormik } from 'formik';
import axios from 'axios';
import React, { useState } from 'react';

function App() {
  const [resultText, setresultText] = useState(
    'To simplify text, type something in the other box and submit!\nTo evaluate the model, no need to type anything :)'
  );

  const evaluateModelClick = () => {
    // setresultText('Evaluating Model ...');

    axios
      .get(
        // somehow putting the trailing / causes 404
        // 'https://text-simplifier-api.herokuapp.com/evaluate'
        'http://127.0.0.1:5000/evaluate'
      )
      .then((response) => {
        console.log('data: ', response.data);
        setresultText(
          'Loss: ' +
            response.data.loss +
            '\nWeighted Accuracy: ' +
            response.data.accuracy +
            '\nPerplexity Per Symbol: ' +
            response.data.perplexity
        );
      })
      .catch((error) => {
        console.log('error: ', error);
      });
  };

  const formik = useFormik({
    initialValues: {
      inputText: '',
    },
    onSubmit: (values) => {
      console.log('values: ', values);

      setresultText('Loading ...');

      const toSend = values.inputText;

      const config = {
        headers: {
          'Content-Type': 'text/plain',
          // 'Access-Control-Allow-Origin': '*',
        },
      };

      axios
        .post(
          // somehow putting the trailing / causes 404
          // 'https://text-simplifier-api.herokuapp.com/simplify',
          'http://127.0.0.1:5000/simplify',
          toSend,
          config
        )
        .then((response) => {
          console.log('data: ', response.data);
          setresultText(response.data.text);
        })
        .catch((error) => {
          console.log('error: ', error);
        });
    },
  });

  return (
    <div className="App">
      <div className="title-div">Text Simplifier</div>
      <div className="whole-div">
        <form onSubmit={formik.handleSubmit}>
          <label htmlFor="inputText">Input Text</label>
          <textarea
            id="inputText"
            name="inputText"
            // type="textArea"
            onChange={formik.handleChange}
            value={formik.values.inputText}
          />
          <div className="btnDiv">
            <AwesomeButton type="primary">Get Simplified Text</AwesomeButton>{' '}
            <AwesomeButton type="primary" onPress={evaluateModelClick}>
              Evaluate Model
            </AwesomeButton>{' '}
          </div>
        </form>

        <div className="resultDiv">
          <label htmlFor="outputText">Output Text</label>
          <textarea
            id="outputText"
            name="outputText"
            // type="textArea"
            // onChange={}
            disabled="true"
            value={resultText}
          />
        </div>
      </div>
      <div className="general-buttons-div">
        <AwesomeButton
          // className="tb-button"
          type="primary"
          href="https://tensorboard.dev/experiment/kGF17AQUQyGUMmCoB62Kpw/#scalars"
          target="_blank"
        >
          Launch Tensorboard
        </AwesomeButton>
      </div>
    </div>
  );
}

export default App;
