#!/usr/bin/env python3
"""
Webservice for sentiment and bot detection keras models

POST to /sentiments for sentiment predictions
POST to /bot_probabilities

both endpoints expect a json with the following format as input:

{
    "texts": [
        {
            "text": "test1"
        },
        {
            "text": "test2"
        }
    ]
}

texts are truncated to 200 chars
"""
import string
import pickle
import textwrap
import logging

from flask import Flask, request, jsonify, abort
from keras import models, preprocessing
from nltk.tokenize import TweetTokenizer
import tensorflow as tf

# declare constants
SENTIMENT_MODEL = '../models/CNN-LSTM/CNN-LSTM_final.h5'
BOT_MODEL = '../models/bot_detection_LSTM/bot_detection_LSTM_final.h5'
SENTIMENT_TOKENIZER = '../models/CNN-LSTM/tokenizer_0.2.pickle'
BOT_TOKENIZER = '../models/bot_detection_LSTM/tokenizer_0.11.pickle'
MAX_WORDS = 50
LOG_LEVEL = 'INFO'
TENSORFLOW_LOG_LEVEL = 'ERROR'


class SentimentService:
    """
    Sentiment service class
    """

    def __init__(self):
        """
        Initialize service object.

        Loads modes and tokenizers from disk

        the _make_predict_function() function is called on both methods to fix
        the issue described in: https://github.com/keras-team/keras/issues/6462
        """

        # configure logging
        logging.basicConfig(level=LOG_LEVEL)
        tf.logging.set_verbosity(TENSORFLOW_LOG_LEVEL)

        logging.info('Loading sentiment analysis model...')
        self.bot_model = models.load_model(BOT_MODEL)
        # pylint: disable=protected-access
        self.bot_model._make_predict_function()

        logging.info('Loading bot detection model...')
        self.sentiment_model = models.load_model(SENTIMENT_MODEL)
        # pylint: disable=protected-access
        self.sentiment_model._make_predict_function()
        self.tweet_tokenizer = TweetTokenizer()
        self.letters = set(string.ascii_lowercase + "'")

        logging.info('Loading sentiment analysis tokenizer...')
        with open(SENTIMENT_TOKENIZER, 'rb') as file:
            self.sentiment_tokenizer = pickle.load(file)

        logging.info('Loading bot detection tokenizer...')
        with open(BOT_TOKENIZER, 'rb') as file:
            self.bot_tokenizer = pickle.load(file)

    def sentiments(self, texts):
        """
        Generate sentiments for a list of texts

        :param texts: list of texts to generate sentiments for
        :return: a dictionary containing input texts and sentiments
        """

        sentiment_dict = self._gen_predictions(texts,
                                               self._predict_sentiments,
                                               'sentiment')

        return sentiment_dict

    def bot_probabilities(self, texts):
        """
        Generate bot probabilities for a list of texts

        :param texts: list of texts to generate bot probabilities for
        :return: a dictionary containing input texts and bot probabilities
        """

        bot_probas_dict = self._gen_predictions(texts,
                                                self._predict_bot_probas,
                                                'bot_probability')

        return bot_probas_dict

    def _gen_predictions(self, texts, pred_function, pred_type):
        """
        Clean up texts and create a response dictionary from passed function

        :param pred_function: function to use for prediction
        :param pred_type: type of prediction: sentiment or bot_probability
        :return: dictionary with input texts and predictions
        """
        truncated_texts = []
        processed_texts = []

        for text in texts:
            incoming_text = textwrap.shorten(text['text'],
                                             width=200,
                                             placeholder="...")
            truncated_texts.append(incoming_text)
            processed_texts.append(self._prepare_text(incoming_text))

        predictions = pred_function(processed_texts)
        pred_list = []
        for text, prediction in zip(truncated_texts, predictions):
            pred_str = '{:.2%}'.format(prediction[0])
            pred_list.append({'text': text, pred_type: pred_str})

        return pred_list

    def _prepare_text(self, text):
        """
        take a text message and prepare it for the models

        cleaning consists of the following operations:

        * convert to lower case
        * convert all URLs to an 'http' tag
        * remove all special characters and all non ascii letters

        :param text: text to prepare
        :return cleaned text
        """

        text = text.lower()
        tokens = self.tweet_tokenizer.tokenize(text)
        tokens = ['http' if t.startswith('http') else t for t in tokens]
        tokens = list(filter(lambda t: set(t).issubset(self.letters), tokens))
        tokens = list(filter(lambda t: not t == "'", tokens))

        return " ".join(tokens)

    def _predict_sentiments(self, texts):
        """
        predict sentiment for list of cleaned texts

        :param texts: list of texts to predict sentiment for
        :return: list of predictions
        """

        sequences = self.sentiment_tokenizer.texts_to_sequences(texts)
        word_vectors = preprocessing.sequence.pad_sequences(sequences,
                                                            maxlen=MAX_WORDS)
        predictions = self.sentiment_model.predict(word_vectors)
        return predictions

    def _predict_bot_probas(self, texts):
        """
        predict bot probabilities for list of cleaned texts

        :param texts: list of texts to predict bot probabilities for
        :return: list of predictions
        """

        sequences = self.bot_tokenizer.texts_to_sequences(texts)
        word_vectors = preprocessing.sequence.pad_sequences(sequences,
                                                            maxlen=MAX_WORDS)
        predictions = self.bot_model.predict(word_vectors)
        return predictions


# initialize flask and sentiment service
app = Flask(__name__)  # pylint: disable=invalid-name
sentiment_service = SentimentService()  # pylint: disable=invalid-name


@app.route('/sentiments', methods=['POST'])
def sentiments():
    """
    sentiments POST endpoint

    :return: returns a json object with the generated sentiments
    """
    if request.method == "POST" and request.is_json is True:
        payload = request.get_json()
        texts = payload['texts']
        pred_list = None

        try:
            pred_list = sentiment_service.sentiments(texts)
        except KeyError:
            abort(422)
    else:
        return abort(400)

    return jsonify({'sentiments': pred_list})


@app.route('/bot_probabilities', methods=['POST'])
def bots():
    """
    bot probabilities POST endpoint

    :return: returns a json object with the generated bot probabilities
    """
    if request.method == "POST" and request.is_json is True:
        payload = request.get_json()
        texts = payload['texts']
        pred_list = None

        try:
            pred_list = sentiment_service.bot_probabilities(texts)
        except KeyError:
            abort(422)
    else:
        return abort(400)

    return jsonify({'bot_probabilities': pred_list})


if __name__ == '__main__':
    logging.info('*** Starting SentiDeep Webservice ***')
    app.run(host='0.0.0.0')
