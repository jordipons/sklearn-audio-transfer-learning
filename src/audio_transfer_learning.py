import os
import random
import numpy as np
from math import ceil
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

import vggish_input, vggish_slim, vggish_params, utils
from utils import wavefile_to_waveform

try:
    import openl3
except:
    print('Warning: you did not install openl3, you cannot use this feature extractor')


DATA_FOLDER = '../data/'
config = {
    'dataset': 'GTZAN',
    'num_classes_dataset': 10,
    'audio_folder': DATA_FOLDER + 'audio/GTZAN/genres/',
    'audio_paths_train': DATA_FOLDER + 'index/GTZAN/train_filtered.txt',
    'audio_paths_test': DATA_FOLDER + 'index/GTZAN/test_filtered.txt',
    'batch_size': 8,
    'features_type': 'vggish', # 'vggish' or 'openl3'
    'model_type': 'linearSVM', # 'linearSVM', 'SVM', 'perceptron', 'MLP', 'kNN'
    # Data: False to compute features or load pre-computed using e.g. 'training_data_GTZAN_vggish.npz'
    'load_training_data': 'training_data_GTZAN_vggish.npz', # False or 'training_data_GTZAN_vggish.npz', 
    'load_evaluation_data': 'evaluation_data_GTZAN_vggish.npz' # False or 'evaluation_data_GTZAN_vggish.npz'
}


def define_classification_model():
    """ Select and define the model you will use for the classifier. 
    """
    if config['model_type'] == 'linearSVM': # linearSVM can be faster than SVM
        return LinearSVC(C=1)
    elif config['model_type'] == 'SVM': # non-linearSVM, we can use the kernel trick
        return SVC(C=1, kernel='rbf', gamma='scale')
    elif config['model_type'] == 'kNN': # k-nearest neighbour
        return KNeighborsClassifier(n_neighbors=1, metric='cosine')
    elif config['model_type'] == 'perceptron': # otpimizes log-loss, also known as cross-entropy with sgd
        return SGDClassifier(max_iter=600, verbose=0.5, loss='log', learning_rate='optimal')
    elif config['model_type'] == 'MLP': # otpimizes log-loss, also known as cross-entropy with sgd
        return MLPClassifier(hidden_layer_sizes=(20,), max_iter=600, verbose=10, 
               solver='sgd', learning_rate='constant', learning_rate_init=0.001)

    
def extract_vggish_features(paths, path2gt): 
    """Extracts VGGish features and their corresponding ground_truth and identifiers (the path).

       VGGish features are extracted from non-overlapping audio patches of 0.96 seconds, 
       where each audio patch covers 64 mel bands and 96 frames of 10 ms each.

       We repeat ground_truth and identifiers to fit the number of extracted VGGish features.
    """
    # 1) Extract log-mel spectrograms
    first_audio = True
    for p in paths:
        if first_audio:
            input_data = vggish_input.wavfile_to_examples(config['audio_folder'] + p)
            ground_truth = np.repeat(path2gt[p], input_data.shape[0], axis=0)
            identifiers = np.repeat(p, input_data.shape[0], axis=0)
            first_audio = False
        else:
            tmp_in = vggish_input.wavfile_to_examples(config['audio_folder'] + p)
            input_data = np.concatenate((input_data, tmp_in), axis=0)
            tmp_gt = np.repeat(path2gt[p], tmp_in.shape[0], axis=0)
            ground_truth = np.concatenate((ground_truth, tmp_gt), axis=0)
            tmp_id = np.repeat(p, tmp_in.shape[0], axis=0)
            identifiers = np.concatenate((identifiers, tmp_id), axis=0)

    # 2) Load Tensorflow model to extract VGGish features
    with tf.Graph().as_default(), tf.Session() as sess:
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, 'vggish_model.ckpt')
        features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
        extracted_feat = sess.run([embedding_tensor], feed_dict={features_tensor: input_data})
        feature = np.squeeze(np.asarray(extracted_feat))

    return [feature, ground_truth, identifiers]


def extract_openl3_features(paths, path2gt):
    """Extracts OpenL3 features and their corresponding ground_truth and identifiers (the path).

       OpenL3 features are extracted from non-overlapping audio patches of 1 second, 
       where each audio patch covers 128 mel bands.

       We repeat ground_truth and identifiers to fit the number of extracted OpenL3 features.
    """
    model = openl3.models.load_embedding_model(input_repr="mel128", 
                                               content_type="music",
                                               embedding_size=512)
    first_audio = True
    for p in paths:
        wave, sr = wavefile_to_waveform(config['audio_folder'] + p, 'openl3')
        emb, _ = openl3.get_embedding(wave, sr, hop_size=1, model=model, verbose=False)
        if first_audio:
            features = emb
            ground_truth = np.repeat(path2gt[p], features.shape[0], axis=0)
            identifiers = np.repeat(p, features.shape[0], axis=0)
            first_audio = False
        else:
            features = np.concatenate((features, emb), axis=0)
            tmp_gt = np.repeat(path2gt[p], emb.shape[0], axis=0)
            ground_truth = np.concatenate((ground_truth, tmp_gt), axis=0)
            tmp_id = np.repeat(p, emb.shape[0], axis=0)
            identifiers = np.concatenate((identifiers, tmp_id), axis=0)

    return [features, ground_truth, identifiers]

    
def extract_features_wrapper(paths, path2gt, model='vggish', save_as=False):
    """Wrapper function for extracting features (VGGish or OpenL3) per batch.
       If a save_as string argument is passed, the features wiil be saved in 
       the specified file.
    """
    if model == 'vggish':
        feature_extractor = extract_vggish_features
    elif model == 'openl3':
        feature_extractor = extract_openl3_features
    else:
        raise NotImplementedError('Current implementation only supports VGGish and OpenL3 features')

    batch_size = config['batch_size']
    first_batch = True
    for batch_id in tqdm(range(ceil(len(paths)/batch_size))):
        batch_paths = paths[(batch_id)*batch_size:(batch_id+1)*batch_size]
        [x, y, refs] = feature_extractor(batch_paths, path2gt)
        if first_batch:
            [X, Y, IDS] = [x, y, refs]
            first_batch = False
        else:
            X = np.concatenate((X, x), axis=0)
            Y = np.concatenate((Y, y), axis=0)
            IDS = np.concatenate((IDS, refs), axis=0)
    
    if save_as:  # save data to file
        # create a directory where to store the extracted training features
        audio_representations_folder = DATA_FOLDER + 'audio_representations/'
        if not os.path.exists(audio_representations_folder):
            os.makedirs(audio_representations_folder)
        np.savez(audio_representations_folder + save_as, X=X, Y=Y, IDS=IDS)
        print('Audio features stored: ', save_as)

    return [X, Y, IDS]


if __name__ == '__main__':

    # load train/test audio paths & ground truth variables
    [paths_train, path2gt_train, path2onehot_train] = utils.load_path2gt(config['audio_paths_train'], config)
    [paths_test, path2gt_test, path2onehot_test] = utils.load_path2gt(config['audio_paths_test'], config)
    paths_all = paths_train + paths_test
    print('Train examples: ' + str(len(paths_train)))
    print('Test examples: ' + str(len(paths_test)))
    print(config)

    if config['load_training_data']:
        print('Loading training features..')
        training_data = np.load(DATA_FOLDER + 'audio_representations/' + config['load_training_data'])
        [X, Y, IDS] = [training_data['X'], training_data['Y'], training_data['IDS']]

    else:
        print('Extracting training features..')
        [X, Y, IDS] = extract_features_wrapper(paths_train, path2gt_train, model=config['features_type'], 
                                               save_as='training_data_{}_{}'.format(config['dataset'], config['features_type']))

    print(X.shape)
    print(Y.shape)

    #interval = range(0, len(X), int(len(X)/250))
    #print(interval)
    #utils.matrix_visualization(X[interval])

    #from sklearn import decomposition
    #pca = decomposition.PCA(n_components=128, whiten=True)
    #pca.fit(X)
    #X = pca.transform(X)
    #print("Shape after PCA: ", X.shape)

    #utils.matrix_visualization(X[interval])

    print('Fitting model..')
    model = define_classification_model()
    model.fit(X, Y)

    print('Evaluating model..')

    if config['load_evaluation_data']:
        print('Loading evaluation features..')
        evaluation_data = np.load(DATA_FOLDER + 'audio_representations/' + config['load_evaluation_data'])
        [X, IDS] = [evaluation_data['X'], evaluation_data['IDS']]

    else:
        print('Extracting evaluation features..')
        [X, Y, IDS] = extract_features_wrapper(paths_test, path2gt_test, model=config['features_type'], 
                                               save_as='evaluation_data_{}_{}'.format(config['dataset'], config['features_type']))

    #X = pca.transform(X)
    #print("Shape after PCA: ", X.shape)

    print('Predict labels on evaluation data')
    pred = model.predict(X)

    # agreggating same ID: majority voting
    y_pred = []
    y_true = []
    for pt in paths_test:
        y_pred.append(np.argmax(np.bincount(pred[np.where(IDS==pt)]))) # majority voting
        y_true.append(int(path2gt_test[pt]))

    # print and store the results
    conf_matrix = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    experiments_folder = DATA_FOLDER + 'experiments/'
    if not os.path.exists(experiments_folder):
        os.makedirs(experiments_folder)
    results_file_name = 'results_{}_{}_{}_{}.txt'.format(config['dataset'],config['features_type'],config['model_type'],random.randint(0,10000))
    to = open(experiments_folder + results_file_name, 'w')
    to.write(str(config) + '\n')
    to.write(str(conf_matrix) + '\n')
    to.write('Accuracy: ' + str(acc))
    to.close()
    print(config)
    print(conf_matrix)    
    print('Accuracy: ' + str(acc))
