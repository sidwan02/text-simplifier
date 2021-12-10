## Results

Adam LR = 0.005

Embedding Size = 50

Train Weighted Sum Accuracy: 0.8028

Train Final Loss: 15949

Train Perplexity Per Symbol: 3.1045

Test Weighted Sum Accuracy: 0.8105

Test Final Loss: 14556

Test Perplexity Per Symbol: 2.8120

## Run Locally

Frontend:

1. cd frontend/
2. yarn start

Backend:

1. cd backend/code/
2. python app.py

## Resources:

1. https://www.tensorflow.org/tensorboard/get_started
2. https://www.tensorflow.org/api_docs/python/tf/data/Dataset
3. https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit

4. https://colab.research.google.com/github/borundev/ml_cookbook/blob/master/Custom%20Metric%20(Confusion%20Matrix)%20and%20train_step%20method.ipynb#scrollTo=cvJxvGHBSoHH
5. https://neptune.ai/blog/keras-metrics

6. https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams
7. https://github.com/tensorflow/tensorboard/issues/2348

8. https://stackoverflow.com/questions/51806852/cant-save-custom-subclassed-model
9. https://machinelearningmastery.com/save-load-keras-deep-learning-models/
10. https://github.com/tensorflow/tensorflow/issues/27688#issuecomment-595950270

11. https://stackabuse.com/deploying-a-flask-application-to-heroku/
12. https://stackoverflow.com/questions/26595874/heroku-src-refspec-master-does-not-match-any
13. https://stackoverflow.com/questions/61062303/deploy-python-app-to-heroku-slug-size-too-large

14. https://flask-cors.readthedocs.io/en/3.0.7/
15. https://stackoverflow.com/questions/42168773/how-to-resolve-preflight-is-invalid-redirect-or-redirect-is-not-allowed-for/42172732#42172732

16. https://elements.heroku.com/buildpacks/heroku/heroku-buildpack-multi-procfile
17. https://github.com/mars/create-react-app-buildpack#user-content-generate-a-react-app

<!-- somehow there isn't a need to cd for the frontend procfile probably because of the heroku prebuild causing it to already be cd into frontend -->

18. https://www.tensorflow.org/api_docs/python/tf/data/experimental/save
19. https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file/32216025
