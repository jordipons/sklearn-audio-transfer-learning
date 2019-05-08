# Audio transfer learning with Scikit-learn

We use the VGGish pre-trained model as a feature extractor. It was trained with Audioset, a dataset conformed by 2M YouTube audios that was designed to train general audio tagging models. On top of it, we simply build a scikit-learn classifier to rapidly prototype competent audio classifiers (even, with few training instances).

See the pipeline:

IMAGE

## Setup with an example

#### Installation:
Create a python 3 virtual environment and install dependencies `pip install -r requirements.txt`

Install tensorflow for CPU `pip install tensorflow` or for CUDA-enabled GPU `pip install tensorflow-gpu`

Be sure to have `ffmpeg` installed, as well.

- **GTZAN fault-filtered version**: download the data [(link)](http://marsyasweb.appspot.com/download/data_sets/). Download the (.txt) files that list which audios are in every partition [(link)](https://github.com/jongpillee/music_dataset_split/tree/master/GTZAN_split). Set the config file: `'train_set_list': 'path/train_audios.txt'`, `'val_set_list': 'path/val_audios.txt'`, `'test_set_list': 'path/test_audios.txt'`, `'audios_list': False`.

## Additional datasets:
Download [US8K dataset](https://urbansounddataset.weebly.com/urbansound8k.html), and ASC-TUT dataset ([dev-set](https://zenodo.org/record/400515#.W9n2UtGdZhE) / [eval-set](https://zenodo.org/record/1040168#.W9n2jNGdZhE)).

#### Preprocess the data:
To preprocess the data, first set some `config_file.py` variables:
- `DATA_FOLDER`, where you want to store all your intermediate files (see folders structure below).
- `config_preprocess['audio_folder']`, where your dataset is located.

Preprocess the data running `python preprocess.py asc_spec`. Note `asc_spec` config option is defined in `config_file.py`

After running `preprocess.py`, spectrograms are in `../DATA_FOLDER/audio_representation/asc__time-freq/`

_*Warning!*_ Rename `index_0.tsv` to `index.tsv`. This is because this script is parallelizable.

#### Regularized deep learning results:

Set `config_sl` dictionary in `config_file.py`, and run `CUDA_VISIBLE_DEVICES=0 python sl_train.py spec`

Once training is done, the resulting model is stored in `../DATA_FOLDER/experiments/fold_0_1541174334/`

To evaluate the model, run `CUDA_VISIBLE_DEVICES=0 python sl_evaluate.py fold_0_1541174334`

#### Prototypical networks results:

Set `config_proto` dictionary in `config_file.py`, and run `CUDA_VISIBLE_DEVICES=0 python proto.py spec`

#### Transfer learning results:
We also study the effectiveness of transfer learning. For that, we use a VGG model pre-trained with Audioset, a dataset conformed by 2 M YouTube audios. This model is available [online](https://github.com/tensorflow/models/tree/master/research/audioset).

For being able to run transfer learning experiments, you just need to download the [VGGish model checkpoint](https://storage.googleapis.com/audioset/vggish_model.ckpt) `vggish_model.ckpt`, and copy it to `/src`.

Then, you can run `transfer_proto.py` following the same logic as in `proto.py`. Now via setting `config_proto` dictionary in `config_file.py`

And you can also run `transfer_train.py` and `transfer_evaluate.py` following the same logic as in `transfer_sl.py` and `transfer_sl.py`. Now via setting `config_transfer_proto` dictionary in `config_file.py`

## Scripts

**Configuration** and preprocessing scripts:
- `config_file.py`: file with all configurable parameters.
- `preprocess.py`: pre-computes and stores the spectrograms.

Scripts for **regularized deep learning models** experiments:
- `sl_train.py`: run it to train your model. First set `config_sl` in `config_file.py`
- `sl_evaluate.py`: run it to evaluate the previously trained model.
- `models_sl.py`: script where the architectures are defined.

Scripts for **prototypical networks** experiments:
- `proto.py`: run it to reproduce our prototypical networks' results. First set `config_proto` in `config_file.py`
- `models_proto.py`: script where the architectures are defined.

Scripts for **transfer learning** experiments:
- `transfer_train.py`: run it to reproduce our transfer learning (with finetuning) results. First set `config_transfer` in `config_file.py`
- `transfer_evaluate.py`: run it to evaluate the previously trained model.
- `transfer_proto.py`: run it to reproduce our prototypical networks' results. First set `config_transfer_proto` in `config_file.py`

Auxiliar scripts:
- `knn_audioset.py`: run it to reproduce our nearest-neigbour Audioset results.
- `knn_mfcc.py`: run it to reproduce our nearest-neigbour MFCCs results.
- `shared.py`: auxiliar script with shared functions that are used by other scripts.
- `vggish_input.py`,`vggish_params.py`,`vggish_slim.py`,`mel_features.py`,`vggish_model.ckpt`: auxiliar scripts for transfer learning experiments.

## Folders structure

- `/src`: folder containing previous scripts.
- `/aux`: folder containing auxiliar additional scripts. These are used to generate the index files in `/data/index/`.
- `/data`: where all intermediate files (spectrograms, results, etc.) will be stored. 
- `/data/index/`: indexed files containing the correspondences between audio files and their ground truth.

When running previous scripts, the following folders will be created:
- `./data/audio_representation/`: where spectrogram patches are stored.
- `./data/experiments/`: where the results of the experiments are stored.
