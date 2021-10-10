CO3320 Project
==============

Using deep learning and sentiment analysis to detect automated agents on social networks
===============================================================================================

Project files for the CO3320 project  "Using deep learning and 
sentiment analysis to detect automated agents on social networks"

The project uses [jupyter notebook](https://jupyter.org), 
[Keras](https://keras.io) and [TensorFlow](https://www.tensorflow.org) to
develop, train and test sentiment analysis and bot detection models for social
media text messages.

The webservice provided is built with [Flask](http://flask.pocoo.org) and can
be packaged in a [docker](https://www.docker.com) image for distribution.

The folder structure of the project follows the 
[Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
project structure framework.


## Requirements

The project was developed on python 3.7.1 and requires a relatively recent 
version of [python](https://www.python.org/downloads/) and 
[pip](https://pypi.org/project/pip/). 

**NOTE:**
The provided `requirements.txt` file contains the 
normal version of TensorFlow, but I would strongly recommend using the
`tensorflow-gpu` package instead if an Nvidia GPU is available in the 
target system. If this is the case, install with:
`pip install tensorflow-gpu`

## Installation
The required python packages for the project can be installed with:

`pip install -r requirements.txt`

I would recommend using a virtual environment for python. More info can be 
found [here](https://docs.python.org/3/tutorial/venv.html).

## Usage 

First thing to do would be to acquire the datasets used to train and test 
the models. They can be downloaded from the following places:

* [Sentiment140](http://help.sentiment140.com/for-students)
* [STS-Gold](https://github.com/pollockj/world_mood/tree/master/sts_gold_v03)
* [Social Honeypot](http://infolab.tamu.edu/data/)
* [Fake Project](https://botometer.iuni.iu.edu/bot-repository/datasets/cresci-2017/cresci-2017.csv.zip)

After downloading the datasets extract them in: `data/external`

To start jupyter notebook run: `jupyter notebook`

I would recommend enabling the following extensions for jupyter: 
`Table of Contents`, `ExecuteTime` and `Runtools`. The instructions from now on 
will assume that the `Table of Contents` extension was enabled. More info about
the jupyter extensions can be found 
[here](https://github.com/ipython-contrib/jupyter_contrib_nbextensions).

The jupyter notebooks for the project can be found in the `notebooks` folder.

The next step after starting jupyter would be to run the data preparation 
notebooks. They can be found under `notebooks/data_preparation`. 

Adjust the folder paths in Section 1.2 of each notebook there if needed 
(should be fine as is if target os is linux or Mac OS X) and run all 
4 of them (can be done with `run all` from the cell menu). 

After the datasets are processed the models are ready to be trained. They can 
be found under `notebooks/models`. If needed adjust the file paths in Section 1.3.
The notebooks represent the final version of the models that were trained on both
the training and validation set. They can all be executed from top to bottom to
create the models, or if you want to use tensorboard and the other keras callbacks,
comment out the uncommented code in Section 3.5 and uncomment the commented out
code in Sections 3.5, 3.6.1 and 3.6.2. The notebooks are designed in such a way
that after running all cells in Section 1, Sections 2, 3 and 4 can be executed 
independently from each other.

With the models ready, the webservice can be started. Adjust the model and 
tokenizer file paths in the file `src/webservie.py` if needed and run it with 
`cd src && python3 'webservice.py'`

The webservice runs on port 5000 and provides 2 POST endpoints. `/sentiments` for
the sentiment analysis model and `/bot_probabilities` for the social bot 
detection model. Both expect a POST request with json body with the following
format:

```json
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
```

#### Docker

A Dockerfile is provided that can be used to create a docker image for the
webservice, adjust the paths in `Dockerfile` if needed and build the docker 
image with `docker build . -t sentideep`

After the image is ready it can be started with

`docker run -p 5000:5000 sentideep`

A docker image with the already trained models is also available on  
[docker hub](https://hub.docker.com/r/etheralm/sentideep) 
and can be directly downloaded and started with the command 
`docker run -p 5000:5000 etheralm/sentideep`

### Other

There are templates under `notebooks/models/templates`, 1 for a sentiment
analysis model and 1 for bot detection. They can be copied over into the models
folder to create and experiment with new model architectures.

The notebooks in `notebooks/analysis` were used to generate the SentiStrength 
benchmark results for the sentiment analysis models and to do the sentiment 
analysis of the social bot datasets. the `sentistrength_test` notebook uses the
output file from SentiStrength and the 2 sentiment analysis evaluation datasets
to evaluate the results from SentiStrength on the same metrics as the 
deep learning models. `The sentiment_bot_datasets` was used to generate the 
sentiment predictions for the 2 bot datasets and to produce the figures and
tables used in the final report.

The folder `notebooks/models/other` includes several other models that were 
tested, but not included in the project.





 