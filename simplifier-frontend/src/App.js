import './App.css';
import { AwesomeButton } from 'react-awesome-button';
import 'react-awesome-button/dist/styles.css';
import { useFormik } from 'formik';

function App() {
  const formik = useFormik({
    initialValues: {
      inputText: '',
    },
    onSubmit: (values) => {
      console.log(values);
    },
  });

  return (
    <div className="App">
      <AwesomeButton
        type="primary"
        href="https://tinyurl.com/tb-adam-hyperparam-seq2seq/"
        target="_blank"
      >
        Tensorboard
      </AwesomeButton>{' '}
      <form onSubmit={formik.handleSubmit}>
        <label htmlFor="inputText">Input Text</label>
        <input
          id="inputText"
          name="inputText"
          type="text"
          onChange={formik.handleChange}
          value={formik.values.inputText}
        />
        {/* <button type="submit">Submit</button> */}
        <AwesomeButton type="primary">Get Simplified Text</AwesomeButton>{' '}
      </form>
    </div>
  );
}

export default App;
