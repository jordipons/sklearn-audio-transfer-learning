# Audio Transfer Learning with Scikit-learn and Tensorflow
We use the [VGGish pre-trained model](https://github.com/tensorflow/models/tree/master/research/audioset) as an audio feature extractor. This deep convolutional neural network was trained with the [Audioset dataset](https://research.google.com/audioset/), that is conformed by 2M YouTube audios for the task of general audio tagging. Scikit-learn classifiers written in python are employed to rapidly prototype competent audio classifiers that can be trained on a CPU.

See the pipeline:

This material is prepared to learn about tensorflow, scikit-learn, and deep learning in general. Besides, due to the simplicity of scikit-learn, this toolkit can be employed to easily build proof-of-concept models from small training sets. See this [article](https://arxiv.org/abs/1810.10274) to know more on using audio transfer learning in low-data regimes. 

## Installation:
Create a python3 virtual environment `python3 -m venv env`, activate it `source ./env/bin/activate`, and install the dependencies `pip install -r requirements.txt`.

#### Download the the VGGish model:
Download the pre-trained model `wget https://storage.googleapis.com/audioset/vggish_model.ckpt` in `sklearn-audio-transfer-learning/src/`.

## Music genre classification: a toy example

#### Setup the task
As an example, let's download the GTZAN dataset `wget http://opihi.cs.uvic.ca/sound/genres.tar.gz` and extract the audio files `tar -zxvf genres.tar.gz` in `sklearn-audio-transfer-learning/data/audio/GTZAN/`. Approximated download time: between 30 min and an hour. We already provide (fault-filtered) train/test partitions in `sklearn-audio-transfer-learning/data/index/GTZAN/`.

`sklearn_audioset.py` is the main python file. Note that in its top-20 lines you can configure it. It is important that you set your data folder, in my case: `DATA_FOLDER = '/home/jordipons/transfer-learning-tutorial/data/'`.

You can also set some parameters. For example, you can select which sklearn classifier to employ. Sklearn has tones of possibilities! You can easily set your favourite sklearn classifier in `define_classification_model()`. To start, let's select 'linearSVM'. We set it as follows: `LinearSVC(C=1)`.

#### Run your classifier
Open `sklearn-audio-transfer-learning/src/` directory in a terminal and run `python sklearn_audioset.py` on your CPU. Approximated run time in your laptop: 15 min.

Congrats, you have build a music genre classifier! The model we developed achieved 77.24% accuracy in our test set. Interestingly, this basic model can achieve better results than a standard [MFCCs + SVM classifier (53.44%)](https://arxiv.org/abs/1805.00237), and is quite competent when compared to the best result we are aware of [(82.1%)](https://www.mdpi.com/2076-3417/8/1/150).

Can you improve this result? Feel free to modify this code. It is meant for that! 
Keep us updated if you break the state-of-the-art ;)

## Scripts directory
- `sklearn_audioset.py`: main script where we build the audio classifiers with Audioset features and Scikit-learn.
- `utils.py`: auxiliar script with util functions that are used by `sklearn_audioset.py`.
- `vggish_input.py`,`vggish_params.py`,`vggish_slim.py`,`mel_features.py`,`vggish_model.ckpt`: auxiliar scripts to employ the VGGish pre-trained model that delivers the Audioset features.

## Folders structure
- `/src`: folder containing previous scripts.
- `/data`: where all intermediate files (Audioset features, audio, results, etc.) will be stored. 
- `/data/index/`: indexed files containing ground truth annotations and partitions.
- `/data/audio/`: folder where to store the audio datasets.

When running previous scripts, the following folders will be created:
- `./data/audio_representation/`: where the training Audioset features are stored.
- `./data/experiments/`: where the results of the experiments are stored.
