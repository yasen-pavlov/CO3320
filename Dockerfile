FROM python:3

ADD src/webservice.py /app/
ADD models/CNN-LSTM/CNN-LSTM_final.h5 /models/CNN-LSTM/
ADD models/CNN-LSTM/tokenizer_0.2.pickle /models/CNN-LSTM/
ADD models/bot_detection_LSTM/bot_detection_LSTM_final.h5 /models/bot_detection_LSTM/
ADD models/bot_detection_LSTM/tokenizer_0.11.pickle /models/bot_detection_LSTM/

RUN pip install flask keras nltk tensorflow

EXPOSE 5000

ENV TF_CPP_MIN_LOG_LEVEL=2

CMD [ "python", "/app/webservice.py" ]