from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import LSTM, Embedding, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, Input
from optuna.integration import TFKerasPruningCallback

import pickle
import re
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import string
from sklearn.model_selection import train_test_split
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


class DataPipeline:
    def __init__(self, tokenizer=None) -> None:
        self.vocab_size = 30000
        self.train_test = 0.30
        self.max_seqlen = 40
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.tokenizer = tokenizer
        self.stopwords = list(set(nltk.corpus.stopwords.words('english') + ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because",
                                                                            "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during",
                                                                            "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here",
                                                                            "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into",
                                                                            "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or",
                                                                            "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should",
                                                                            "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's",
                                                                            "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up",
                                                                            "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's",
                                                                            "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've",
                                                                            "your", "yours", "yourself", "yourselves"] + list(string.ascii_lowercase)))

    def tokenization_punct(self, text):
        tokens = re.findall(r"[\w]+|[^\s\w]", text)
        return tokens

    def remove_punctuation(self, text):
        punctuationfree = [i for i in text if i not in string.punctuation]
        return punctuationfree

    def remove_stopwords(self, text):
        output = [i for i in text if i not in self.stopwords]
        return output

    def lemmatizer(self, text):
        lemm_text = [self.wordnet_lemmatizer.lemmatize(word) for word in text]
        return " ".join(lemm_text)

    def remove_numbers(self, text):
        result = re.sub(r'\d+', '', text)
        return result

    def split_train_test(self, data, xcols, ycolname):
        data[ycolname] = data[ycolname].map(
            {'NEGATIVE': 0, 'POSITIVE': 1})

        X = data[xcols]
        Y = data[ycolname].values
        # split up the data
        return train_test_split(X, Y, test_size=self.train_test, stratify=Y)

    def tokenize_pad_sentences(self, sentences):
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(
                oov_token='UNK', num_words=self.vocab_size)
            self.tokenizer.fit_on_texts(sentences)
            self.tokenizer.word_index = {e: i for e,
                                         i in self.tokenizer.word_index.items() if i <= self.vocab_size}
            with open('tokenizer.pickle', 'wb') as handle:
                pickle.dump(self.tokenizer, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
        sentences = self.tokenizer.texts_to_sequences(sentences)
        return pad_sequences(sentences, maxlen=self.max_seqlen, padding='pre')


class LSTMmodel:
    def __init__(self, datapipeline) -> None:
        self.datapipeline = datapipeline
        self.glove_index = {}
        self.embedding_dim = 100
        self.embedding_layer = None
        self.fitted_model = None
        self.network = None
        self.epochs = 10
        self.train_batchsize = 512
        self.val_batchsize = 32

    def get_glove_vectors(self):
        wordvectors_path = './glove.6B.100d.txt'
        with open(wordvectors_path) as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                self.glove_index[word] = coefs

    def prepare_embedding_matrix(self):
        hits = 0
        misses = 0
        self.datapipeline.vocab_size = self.datapipeline.vocab_size + 1
        embedding_matrix = np.zeros(
            (self.datapipeline.vocab_size, self.embedding_dim))
        for word, i in self.datapipeline.tokenizer.word_index.items():
            embedding_vector = self.glove_index.get(word)
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                # This includes the representation for "padding" and "OOV"
                embedding_matrix[i] = embedding_vector
                hits += 1
            else:
                misses += 1
        print("Converted %d words (%d misses)" % (hits, misses))

        # Define the embedding layer
        self.embedding_layer = Embedding(input_dim=self.datapipeline.vocab_size,
                                         output_dim=self.embedding_dim,
                                         input_length=self.datapipeline.max_seqlen,
                                         weights=[embedding_matrix],
                                         trainable=False)

    def lstm_model_training(self, trial, Xtrain, Xtest, Ytrain, Ytest):

        inputs = Input(shape=(self.datapipeline.max_seqlen,))
        embeddings = self.embedding_layer(inputs)
        X = LSTM(trial.suggest_categorical(f"num_units_0", [
            32, 64, 128]), return_sequences=True)(embeddings)
        X = Dropout(trial.suggest_float(f"dropout_0", 0.3, 0.5, step=0.1))(X)
        for i in range(trial.suggest_categorical(f"num_layers", [0, 1, 2])):
            X = LSTM(trial.suggest_categorical(
                f"num_units_{i+1}", [32, 64, 128]), return_sequences=True)(X)
            X = Dropout(trial.suggest_float(
                f"dropout_{i+1}", 0.3, 0.5, step=0.1))(X)
        X = LSTM(trial.suggest_categorical(
            f"num_units_last", [32, 64, 128]))(X)
        for j in range(trial.suggest_categorical(f"dense_layers", [0, 1])):
            X = Dense(trial.suggest_categorical(
                f"dense_units", [32, 64, 128]), activation='relu')(X)
        outputs = Dense(1, activation='sigmoid')(X)
        self.network = Model(inputs=inputs, outputs=outputs)

        # We compile our model with a sampled learning rate.
        learning_rate = trial.suggest_float(
            "learning_rate", 1e-5, 1e-1, log=True)

        self.network.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=learning_rate),
            metrics=['accuracy'],
        )

        learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                                    patience=1,
                                                    verbose=1,
                                                    factor=0.5,
                                                    min_lr=1e-5)

        early_stopping = EarlyStopping(
            monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)

        callbacks = [learning_rate_reduction, early_stopping,
                     TFKerasPruningCallback(trial, 'val_loss')]

        self.fitted_model = self.network.fit(Xtrain,
                                             Ytrain.reshape((-1, 1)),
                                             batch_size=self.train_batchsize,
                                             epochs=self.epochs,
                                             shuffle=True,
                                             validation_data=(
                                                 Xtest, Ytest.reshape((-1, 1))),
                                             validation_batch_size=self.val_batchsize,
                                             callbacks=callbacks
                                             )

    def evaluate_results(self, Xtest, Ytest):
        preds = self.network.predict(Xtest, batch_size=self.val_batchsize)
        preds = [1 if i[0] >= 0.5 else 0 for i in preds]

        # Log accuracy, precision and recall metrics
        accuracy = accuracy_score(Ytest.reshape((-1, 1)), preds)
        precision = precision_score(Ytest.reshape((-1, 1)), preds)
        recall = recall_score(Ytest.reshape((-1, 1)), preds)
        f1 = f1_score(Ytest.reshape((-1, 1)), preds)

        return accuracy, precision, recall, f1
