import './App.css';
import { AwesomeButton } from 'react-awesome-button';
import 'react-awesome-button/dist/styles.css';
import { useFormik } from 'formik';
import axios from 'axios';
import React, { useState } from 'react';

function App() {
  const [resultText, setresultText] = useState(
    'Type something in the other box and submit!'
  );

  const formik = useFormik({
    initialValues: {
      inputText: '',
    },
    onSubmit: (values) => {
      console.log('values: ', values);

      setresultText('Generating simplified text ...');

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
          'https://text-simplifier-api.herokuapp.com/evaluate-model',
          toSend,
          config
        )
        .then((response) => {
          console.log('data: ', response.data);
          setresultText(response.data.stdout);
        })
        .catch((error) => {
          console.log('error: ', error);
        });
    },
  });

  return (
    <div className="App">
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
          href="https://tinyurl.com/tb-adam-hyperparam-seq2seq/"
          target="_blank"
        >
          Launch Tensorboard
        </AwesomeButton>
      </div>
    </div>
  );
}

export default App;
