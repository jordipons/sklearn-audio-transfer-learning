# Audio Transfer Learning with Scikit-learn and Tensorflow

We use the [VGGish pre-trained model](https://github.com/tensorflow/models/tree/master/research/audioset) as an audio feature extractor. This deep convolutional neural network was trained with the [Audioset dataset](https://research.google.com/audioset/), that is conformed by 2M YouTube audios for the task of general audio tagging. Scikit-learn classifiers written in python are employed to rapidly prototype competent audio classifiers that can be trained on a CPU.

See the pipeline:

This material is prepared to learn about tensorflow, scikit-learn, and deep learning in general. Besides, due to the simplicity of scikit-learn, this toolkit can be employed to easily build proof-of-concept models from small training sets. See this [article](https://arxiv.org/abs/1810.10274) to know more on using audio transfer learning in low-data regimes. 

## Installation:
Create a python3 virtual environment `python3 -m venv env`, activate it `source ./env/bin/activate` and install the dependencies `pip install -r requirements.txt`.

#### Download the the VGGish model:
Download the pre-trained model `wget https://storage.googleapis.com/audioset/vggish_model.ckpt` in `sklearn-audio-transfer-learning/src/`.

## Music genre classification: a toy example

#### Download the data:

As an example, let's download the GTZAN dataset `wget http://opihi.cs.uvic.ca/sound/genres.tar.gz`and extract the audio files `tar -zxvf genres.tar.gz` in `sklearn-audio-transfer-learning/data/audio/GTZAN/`. Approximated download time: between 30 min and an hour.

`sklearn_audioset.py` is the main python file. You can configure it in its top-20 lines. It is important that you set your data folder, in my case: `DATA_FOLDER = '/home/idrojsnop/transfer-learning-tutorial/data/'`. 

Below, you can also set some parameters. For example, you can select which sklearn models to employ: 'linearSVM', 'SVM', 'MLP', or 'kNN'. But sklearn has tones of possibilities! For that reason, you can easily set your sklearn classifiers in `define_classification_model()`. To start, let's select 'linearSVM' set as follows: `LinearSVC(C=1)`.

Now, run `python src/sklearn_audioset.py` on your CPU.

Congrats! You now know how to build a music genre classifier. The model we developed achieved 77.24% accuracy in our test set. Interestingly, this basic model can achieve better results than a standard [MFCCs + SVM (53.44%)](https://arxiv.org/abs/1805.00237), and is quite close to the best result we are aware of (82.1%).

## Scripts directory
- `knn_audioset.py`: run it to reproduce our nearest-neigbour Audioset results.
- `knn_mfcc.py`: run it to reproduce our nearest-neigbour MFCCs results.
- `shared.py`: auxiliar script with shared functions that are used by other scripts.
- `vggish_input.py`,`vggish_params.py`,`vggish_slim.py`,`mel_features.py`,`vggish_model.ckpt`: auxiliar scripts for transfer learning experiments.

## Folders structure

- `/src`: folder containing previous scripts.
- `/data`: where all intermediate files (spectrograms, results, etc.) will be stored. 
- `/data/index/`: indexed files containing the correspondences between audio files and their ground truth.

When running previous scripts, the following folders will be created:
- `./data/audio_representation/`: where spectrogram patches are stored.
- `./data/experiments/`: where the results of the experiments are stored.


## Additional information:
This material was prepared with didactic intentions. Interested students, might be interested in 

Download [US8K dataset](https://urbansounddataset.weebly.com/urbansound8k.html), and ASC-TUT dataset ([dev-set](https://zenodo.org/record/400515#.W9n2UtGdZhE) / [eval-set](https://zenodo.org/record/1040168#.W9n2jNGdZhE)).


## TODO:
- Store the results.
