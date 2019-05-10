#### Why we get different when running several times the same neural network-based experiment?

For example, with this setup `MLPClassifier(hidden_layer_sizes=(20,), max_iter=600, verbose=10, early_stopping=False)` we get these scores: 77.93%, 73.44%, or 75.12%.

Or, as another example, with this setup `SGDClassifier(max_iter=600, verbose=0.5, loss='log', learning_rate='optimal')` we get these scores: 75.51%, or 75.17%.

#### Some ideas in how to improve your results:
- hyper parameter search
- try different models
- feature selection
- dimensionality reduction
- ensembles
- discuss random initialization and ensembles
- discuss overfitting
- visualize stuff

#### Tips to deploy your best model: 
- https://scikit-learn.org/stable/modules/model_persistence.html

#### Some complementary readings:
- (Training neural audio classifiers with few data)[https://arxiv.org/abs/1810.10274]
