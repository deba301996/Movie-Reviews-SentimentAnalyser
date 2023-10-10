import pandas as pd
import mlflow
import matplotlib.pyplot as plt
import optuna

from datetime import datetime
from optuna.integration.mlflow import MLflowCallback
from data_model_class import DataPipeline, LSTMmodel
from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_intermediate_values
from optuna.visualization.matplotlib import plot_slice
from optuna.trial import TrialState

# Read the data:
movie_reviews = pd.read_csv(
    './rotten_tomatoes_movie_reviews.csv', usecols=['reviewText', 'scoreSentiment'])

# Remove rows without reviews:
movie_reviews = movie_reviews[~movie_reviews['reviewText'].isna()]

# Take only 500,000 rows initially:
movie_reviews = movie_reviews.head(500000)

# Create an instance of the data papeline:
pipeline = DataPipeline()

movie_reviews['reviewText'] = movie_reviews['reviewText'].apply(
    lambda x: x.lower())
movie_reviews['reviewText'] = movie_reviews['reviewText'].apply(
    lambda x: pipeline.tokenization_punct(x))
movie_reviews['reviewText'] = movie_reviews['reviewText'].apply(
    lambda x: pipeline.remove_punctuation(x))
movie_reviews['reviewText'] = movie_reviews['reviewText'].apply(
    lambda x: pipeline.remove_stopwords(x))
movie_reviews['reviewText'] = movie_reviews['reviewText'].apply(
    lambda x: pipeline.lemmatizer(x))
movie_reviews['reviewText'] = movie_reviews['reviewText'].apply(
    lambda x: pipeline.remove_numbers(x))

# Train test split:
Xtrain, Xtest, Ytrain, Ytest = pipeline.split_train_test(
    movie_reviews, 'reviewText', 'scoreSentiment')

# Tokenize and pad sentences:
Xtrain = pipeline.tokenize_pad_sentences(Xtrain)
Xtest = pipeline.tokenize_pad_sentences(Xtest)

model = LSTMmodel(pipeline)
model.get_glove_vectors()
model.prepare_embedding_matrix()

# MLflow starts here
# Get current date and time
now = datetime.now()
formatted_date_time = now.strftime("%Y%m%d%H%M%S")
experiment_name = 'model_tuning_' + formatted_date_time
mlflow.set_experiment(experiment_name)

mlflc = MLflowCallback(
    tracking_uri=mlflow.get_tracking_uri(), create_experiment=False, metric_name="accuracy")

# Define the objective function for Optuna study


@mlflc.track_in_mlflow()
def objective(trial):
    model.lstm_model_training(trial, Xtrain, Xtest, Ytrain, Ytest)
    accuracy, precision, recall, f1 = model.evaluate_results(Xtest, Ytest)

    # Save the model
    trial.set_user_attr(key="model_obj", value=model.network)

    # Log metrics
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Log the accuracy and loss values each epoch as plots:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(model.fitted_model.history['accuracy'], label='accuracy')
    ax1.plot(model.fitted_model.history['val_accuracy'], label='val_accuracy')
    ax1.set_title('Accuracy plot')
    ax1.legend()

    ax2.plot(model.fitted_model.history['loss'], label='loss')
    ax2.plot(model.fitted_model.history['val_loss'], label='val_loss')
    ax2.set_title('Loss plot')
    ax2.legend()
    plt.tight_layout()
    mlflow.log_figure(fig, 'plots/model_training_plots.png')

    return accuracy


def show_result(study):
    pruned_trials = study.get_trials(
        deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(
        deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


study = optuna.create_study(
    direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=2))
study.optimize(objective, n_trials=30,
               show_progress_bar=True, callbacks=[mlflc])
show_result(study)

# Register the best model:
mlflow.end_run()

with mlflow.start_run(run_id=study.best_trial.system_attrs['mlflow_run_id']):
    mlflow.tensorflow.log_model(model=study.best_trial.user_attrs['model_obj'],
                                artifact_path='tf_model',
                                registered_model_name='sentiment_analyzer_model')  # this argument registers the model

# Transition registered model to production:
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="sentiment_analyzer_model",
    version=1,
    stage="Production"
)

# Save the plots
plot_optimization_history(study)
plt.savefig("optimization_history.png")
plt.clf()

plot_param_importances(study)
plt.savefig("param_importances.png")
plt.clf()

plot_intermediate_values(study)
plt.savefig("intermediate_values.png")
plt.clf()

plot_slice(study)
plt.savefig("slice.png")
plt.clf()
