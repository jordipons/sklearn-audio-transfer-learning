#### Quick setup for a class:

Clone this Github repository: `git clone https://github.com/jordipons/sklearn-audio-transfer-learning.git`.

After cloning, access to the repository to build a virtual environment: `cd sklearn-audio-transfer-learning/`. Create a python3 virtual environment `python3 -m venv env`, activate it `source ./env/bin/activate`, and install the dependencies `pip install -r requirements.txt`.

Run the main python script: `cd src/`, and `python audio_transfer_learning.py`.

#### Why we get different results when running several times the same neural network-based experiment?

For example, with this setup `MLPClassifier(hidden_layer_sizes=(20,), max_iter=600, verbose=10, early_stopping=False)` we get these scores: 77.93%, 73.44%, or 75.12%.

Or, as another example, with this setup `SGDClassifier(max_iter=600, verbose=0.5, loss='log', learning_rate='optimal')` we get these scores: 75.51%, or 75.17%.

#### With or without PCA? Overfitting?

*[The following experiments employ Openl3 features]*  

MLPClassifier(hidden_layer_sizes=(128,), max_iter=**10**, verbose=10,
               solver='sgd', learning_rate='constant', learning_rate_init=0.001)
PCA: 72.06%                                      - NO PCA: 73.79 %

MLPClassifier(hidden_layer_sizes=(128,), max_iter=**50**, verbose=10,
               solver='sgd', learning_rate='constant', learning_rate_init=0.001)
PCA: 75.17% / NO PCA: **76.55%**

MLPClassifier(hidden_layer_sizes=(128,), max_iter=**100**, verbose=10,
               solver='sgd', learning_rate='constant', learning_rate_init=0.001)
PCA: **77.24%** / NO PCA: 71.37%

MLPClassifier(hidden_layer_sizes=(128,), max_iter=**300**, verbose=10,
               solver='sgd', learning_rate='constant', learning_rate_init=0.001)
PCA: 75.17% / NO PCA: 74.48%

MLPClassifier(hidden_layer_sizes=(128,), max_iter=**600**, verbose=10,
               solver='sgd', learning_rate='constant', learning_rate_init=0.001)
PCA: 74.48% / NO PCA: 73.79 %

MLPClassifier(hidden_layer_sizes=(128,), max_iter=**1000**, verbose=10,
               solver='sgd', learning_rate='constant', learning_rate_init=0.001)
PCA: 75.86% / NO PCA: 72.75 %  **(confusion matrix below)**

(without PCA the model has more trainable parameters, and the model overfits before!)


#### What are these numbers?  
[[11  0  3  0  1 12  0  0  0  4]
 [ 0 31  0  0  0  0  0  0  0  0]
 [ 0  0 25  4  0  0  0  0  0  1]
 [ 0  0  0 24  3  0  0  1  1  0]
 [ 0  0  0  2 24  0  0  1  0  0]
 [ 1  0  0  0  0 21  0  1  0  4]
 [ 0  0  0  0  0  0 26  0  0  1]
 [ 0  0  0  2  1  0  0 27  0  0]
 [ 0  0  1  3  4  0  0  1 16  1]
 [ 1  0  5  7  3  1  2  4  3  6]]  
 
A confusion matrix!

#### SVM or MLP or KNN? 
What's working the best? What's faster? What's simpler to train?  
Let's engage our students with these practical questions!

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

#### Tips to deploy your best model: 
- https://scikit-learn.org/stable/modules/model_persistence.html

#### Some complementary readings:
- [Training neural audio classifiers with few data](https://arxiv.org/abs/1810.10274)
