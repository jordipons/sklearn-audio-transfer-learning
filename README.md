# Audio Transfer Learning with Scikit-learn

We use the [VGGish pre-trained model](https://github.com/tensorflow/models/tree/master/research/audioset) as a feature extractor. It was trained with Audioset, a dataset conformed by 2M YouTube audios that was designed to train general audio tagging models. On top of it, we simply build a scikit-learn classifier to rapidly prototype competent audio classifiers (even, with few training instances).

See the pipeline:

IMAGE

## Installation:
Create a python 3 virtual environment and install dependencies `pip install -r requirements.txt`

Install tensorflow for CPU `pip install tensorflow` or for CUDA-enabled GPU `pip install tensorflow-gpu`

Be sure to have `ffmpeg` installed, as well.

#### Download the the VGGish model:
Download the pre-trained model `wget https://storage.googleapis.com/audioset/vggish_model.ckpt` in `sklearn-audio-transfer-learning/src/`.

## A toy example: genre classification

#### Download the data:

As an example, let's download the **GTZAN** dataset `wget http://opihi.cs.uvic.ca/sound/genres.tar.gz`and extract the audio files `tar -zxvf genres.tar.gz` in `sklearn-audio-transfer-learning/data/audio/GTZAN/`.

And run `python sklearn_audioset.py`.

Note that in line ? one can configure the script, and in lines ? one can set several sklearn classifiers. For example, with this setup we get that ? accuracy. 

This can be run in a CPU. Didactic porpuses.

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
