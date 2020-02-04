# Audio Transfer Learning with Scikit-learn and Tensorflow
We use pre-trained Tensorflow models as audio feature extractors, and Scikit-learn classifiers are employed to rapidly prototype competent audio classifiers that can be trained on a CPU.

See the pipeline when using the [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset) model (but note you can also use [MusiCNN](https://github.com/jordipons/musicnn) and [Openl3](https://github.com/marl/openl3)):

<p align="center"><img src="sklearn-audio-transfer-learning.png"  height="65"></p>

This material was prepared to teach Tensorflow, Scikit-learn, and deep learning in general. Besides, due to the simplicity of Scikit-learn, this toolkit can be employed to easily build proof-of-concept models with your own data.

## Installation:

Clone this repository: `git clone https://github.com/jordipons/sklearn-audio-transfer-learning.git`.

Create a python3 virtual environment `python3 -m venv env`, activate it `source ./env/bin/activate`, and install the dependencies `pip install -r requirements.txt`.

Download the pre-trained models you want to use as feature extractors. Our current implementation supports:

- **VGGish:** download the pre-trained model `wget https://storage.googleapis.com/audioset/vggish_model.ckpt` in `sklearn-audio-transfer-learning/src/`. For more details, check out their [documentation](https://github.com/tensorflow/models/tree/master/research/audioset).
- **OpenL3:** download it via pip `pip install --timeout 100000000 openl3 `. Use a large timeout, because it takes some time to download the model. For more details, check out their [documentation](https://github.com/marl/openl3).
- **MusiCNN:** download it via pip `pip install musicnn`. For more details, chec out their [documentation](https://github.com/jordipons/musicnn).

## Music genre classification: a toy example

#### Set the task up
As an example, let's download the GTZAN dataset `wget http://opihi.cs.uvic.ca/sound/genres.tar.gz` and extract the audio files `tar -zxvf genres.tar.gz` in `sklearn-audio-transfer-learning/data/audio/GTZAN/`. Approximated download time: between 30 min and an hour. We already provide (fault-filtered) train/test partitions in `sklearn-audio-transfer-learning/data/index/GTZAN/`.

`audio_transfer_learning.py` is the main python file. Note that on its top-20 lines you can configure it. For example: you can select *(i)* which sklearn classifier to employ, and *(ii)* which pre-trained model to use for extracting features.  

You can easily set your favourite sklearn classifier in `define_classification_model()`. To start, let's select `SVM`. We set it as follows: `SVC(C=1, kernel='rbf', gamma='scale')`. Additionally, we employ [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) dimensionality reduction via setting `'pca': 128`. Finally, we select which pre-trained Tensorflow model to use as feature extactor. To start, let's use the `vggish`. Remember to download the pre-trained model first!

#### Run your classifier
Open the `sklearn-audio-transfer-learning/src/` directory in a terminal and run `python audio_transfer_learning.py`.

Congrats, you have build a music genre classifier! The model we developed (VGGish + 128 PCA + SVM) achieved 77.58% accuracy in our test set. Employing this same setup with `openl3` features, we achieve (Openl3 + 128 PCA + SVM): 74.65% accuracy -- and with `musicnn` (max_pool) features, we achieve (musicnn (max_pool) + 128 PCA + SVM): 77.24% accuracy. Interestingly, these basic models can achieve better results than a standard [MFCCs + SVM classifier (53.44%)](https://arxiv.org/abs/1805.00237), and are quite competent when compared to the best result we are aware of: [82.1% accuracy](https://www.mdpi.com/2076-3417/8/1/150).

#### Can you improve this result? 

Feel free to modify our scripts, they are meant for that! 
Keep us updated if you break the state-of-the-art ;)

## I want to build my own audio classifier

In the following, we give some tips on how to build another audio classifier that is not based on the GTZAN dataset.

As an example on how to do it, let's download the ASC-TUT dataset ([dev-set](https://zenodo.org/record/400515#.W9n2UtGdZhE) / [eval-set](https://zenodo.org/record/1040168#.W9n2jNGdZhE)). With this data, you can build an acoustic scene classifier.

Copy the audio into a new directory: `sklearn-audio-transfer-learning/data/audio/ASC-TUT/`. Now create your train/test partitions. These are just lists of the files that belong to this partition. For example, access to your audio directory `sklearn-audio-transfer-learning/data/audio/ASC-TUT/dev-set/` and run `ls > train_set.txt`. Do the same for creating the test partition. Remember to configure the variables `audio_paths_train` and `audio_paths_test` (in `audio_transfer_learning.py`) with the new dataset/partitions.

The last step is to define which is the label (or ground truth) for each of the audios. You can define the correspondence between the audio path and its label in the following function `path2gt_datasets(path, dataset)` in `sklearn-audio-transfer-learning/src/utils.py`.

You are ready to go!

## Additional tips
    
#### Do you want to experiment with different Scikit-learn models?

You can easily set your favourite sklearn classifier in `define_classification_model()`. Then, select one `model_type`. The following classifiers are already implemented: `linearSVM`, `SVM`, `perceptron`, `MLP`, and `kNN`. Check [Scikit-learn's documentation](https://scikit-learn.org/stable/) to know more about its possibilities.
    
#### Does the script takes an eternity to extract the features?

Once you have extracted the features once, these are automatically stored in `sklearn-audio-transfer-learning/data/audio_representation/`. You can load those (instead of re-computing) by simply setting the variable `load_training_data` with the name of the file containing the pre-computed training features (e.g.: `evaluation_data_GTZAN_vggish.npz`).

To re-compute the features, just set `load_training_data` to False.

## Scripts directory
- `audio_transfer_learning.py`: main script where we build the audio classifiers with Tensorflow and Scikit-learn.
- `utils.py`: auxiliar script with util functions that are used by `audio_transfer_learning.py`.
- `vggish_input.py`,`vggish_params.py`,`vggish_slim.py`,`mel_features.py`,`vggish_model.ckpt`: auxiliar scripts to employ the VGGish pre-trained model.

## Folders structure
- `/src`: folder containing the previously listed scripts.
- `/data`: where all intermediate files (pre-computed features, audio, results, etc.) will be stored. 
- `/data/audio/`: recommended folder where to store the audio datasets.
- `/data/index/`: recommended folder where to store the indexed files containing ground truth annotations and partitions.

When running these scripts, the following folders will be created:
- `./data/audio_representation/`: where the pre-computed features are stored.
- `./data/experiments/`: where the results of the experiments are stored

## Are you using sklearn-audio-transfer-learning?
If you are using it for academic works, please cite us:
```
@inproceedings{pons2018atscale,
  title={End-to-end learning for music audio tagging at scale},
  author={Pons, Jordi and Nieto, Oriol and Prockup, Matthew and Schmidt, Erik M. and Ehmann, Andreas F. and Serra, Xavier},
  booktitle={19th International Society for Music Information Retrieval Conference (ISMIR2018)},
  year={2018},
}

```
```
@inproceedings{pons2019musicnn,
  title={musicnn: pre-trained convolutional neural networks for music audio tagging},
  author={Pons, Jordi and Serra, Xavier},
  booktitle={Late-breaking/demo session in 20th International Society for Music Information Retrieval Conference (LBD-ISMIR2019)},
  year={2019},
}

```
If you use it for other purposes, let us know!
