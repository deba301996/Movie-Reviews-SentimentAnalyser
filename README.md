# Movie Reviews Sentiment Analyzer

The salient features of the project are:
+ Tensorflow Keras based **LSTM** model utilizing **GloVe embeddings**
+ Hyperparameter tuning using **Optuna**
+ Tracking metrics and artifacts of runs using **MLflow**
+ Model serving and productionalizing using **MLflow**
+ Deploying model using **FastAPI**

Each of the files in the repository are described below:
+ **training_tracking.py** : Main script for model training and running the MLflow experiment
+ **data_model_class.py** : Contains class for data preprocessing and model network
+ **deployment.py** : FastAPI model deployment script
+ **requirements.txt** : Conda environment and package dependencies
+ **sample_test.csv** : Sample file that can be uploaded to generate predictions
+ **tokenizer.pickle** : Saved Keras tokenizer object
+ **mlruns/** : Contains MLflow runs 

To use the trained model for generating predictions, follow the steps listed below:

1. Clone the repository in a new directory using the command
```
git clone https://github.com/deba301996/Movie-Reviews-SentimentAnalyser.git
```
2. Create a new conda virtual environment using the **requirements.txt** file using the command
```
conda create --name new_environment_name --file requirements.txt
```
3. Activate the conda environment
```
conda activate new_environment_name
```
4. Start the MLflow tracking server by running the following command from the terminal. The tracking server by default will start on **http://127.0.0.1:5000**
```
mlflow ui
```
5. In a different terminal window, activate the conda environment and change to the working directory. Post that, set the MLflow tracking uri using
```
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```
6. Next, start the model serving server on a different port 1234 using the command. It will start the server on **http://127.0.0.1:1234**
```
mlflow models serve --model-uri models:/sentiment_analyzer_model
/Production -p 1234 --no-conda
```
7. Lastly, run the **deployment.py** script using the command, which opens on **http://127.0.0.1:8000**
```
uvicorn deployment:app --reload
```
8. Go to the url **http://127.0.0.1:8000/docs#/** where there are 2 options to generate predictions - either enter a single movie review, or upload a file similar to the **sample_test.csv** to get the batch predictions.