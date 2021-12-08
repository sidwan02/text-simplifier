import './App.css';
import { AwesomeButton } from 'react-awesome-button';
import 'react-awesome-button/dist/styles.css';
import { useFormik } from 'formik';
import axios from 'axios';

function App() {
  const formik = useFormik({
    initialValues: {
      inputText: '',
    },
    onSubmit: (values) => {
      console.log('values: ', values);

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
          {/* <button type="submit">Submit</button> */}
          <div className="btnDiv">
            <AwesomeButton type="primary">Get Simplified Text</AwesomeButton>{' '}
            <AwesomeButton
              // className="tb-button"
              type="primary"
              href="https://tinyurl.com/tb-adam-hyperparam-seq2seq/"
              target="_blank"
            >
              Launch Tensorboard
            </AwesomeButton>
          </div>
        </form>
        <div className="resultDiv">
          <div>Output Text</div>
          <div className="simplified-text-div"></div>
        </div>
      </div>
    </div>
  );
}

export default App;
