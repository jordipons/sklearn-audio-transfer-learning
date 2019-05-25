#### Quick setup for a class:

Clone this Github repository: `git clone https://github.com/jordipons/sklearn-audio-transfer-learning.git`.

Create a python3 virtual environment `python3 -m venv env`, activate it `source ./env/bin/activate`, and install the dependencies `pip install -r requirements.txt`.

Run the main python script: `cd src/`, and `python audio_transfer_learning.py`.

#### Why we get different results when running several times the same neural network-based experiment?

For example, with this setup `MLPClassifier(hidden_layer_sizes=(20,), max_iter=600, verbose=10, early_stopping=False)` we get these scores: 77.93%, 73.44%, or 75.12%.

Or, as another example, with this setup `SGDClassifier(max_iter=600, verbose=0.5, loss='log', learning_rate='optimal')` we get these scores: 75.51%, or 75.17%.

#### Some ideas in how to improve your results:
- Try different models
- Hyper-parameter search
- Try different features
- Visualize parts of the model
- Dimensionality reduction
- Discuss overfitting
- Feature selection
- Ensembles
- tSNE of the data and features

Share your results in: https://docs.google.com/document/d/1oyqFCy9P-BD13b2FyWMkTdxbdWPfNRP6ATexqvbw4xA/edit

#### Tips to deploy your best model: 
- https://scikit-learn.org/stable/modules/model_persistence.html

#### Some complementary readings:
- [Training neural audio classifiers with few data](https://arxiv.org/abs/1810.10274)
