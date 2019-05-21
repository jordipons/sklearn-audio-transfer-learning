import os
import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import vggish_input, vggish_slim, vggish_params, utils

DATA_FOLDER = '/home/idrojsnop/Dropbox/Dolby/sklearn-audio-transfer-learning/data/'
config = {
    'dataset': 'GTZAN',
    'num_classes_dataset': 10,
    'audio_folder': DATA_FOLDER + 'audio/GTZAN/genres/',
    'audio_paths_train': DATA_FOLDER + 'index/GTZAN/train_filtered.txt',
    'audio_paths_test': DATA_FOLDER + 'index/GTZAN/test_filtered.txt',
    'train_batch': 8,
    'test_batch': 8,
    'model_type': 'linearSVM', # 'linearSVM', 'SVM', 'perceptron', 'MLP', 'kNN'
    'load_training_data': 'training_data_GTZAN_8643.npz' # False or load a model: 'training_data_GTZAN_839.npz'
}

def define_classification_model():
    """ Select and define the model you will use for the classifier. 
    """
    if config['model_type'] == 'linearSVM':
        # linearSVM can be faster than SVM
        from sklearn.svm import LinearSVC
        return LinearSVC(C=1)
    elif config['model_type'] == 'SVM':
        from sklearn.svm import SVC
        return SVC(C=1, kernel='rbf', gamma='scale')
    elif config['model_type'] == 'perceptron':
        # otpimizes log-loss, also known as cross-entropy with sgd
        from sklearn.linear_model import SGDClassifier
        return SGDClassifier(max_iter=600, verbose=0.5, loss='log', learning_rate='optimal')
    elif config['model_type'] == 'MLP':
        # otpimizes log-loss, also known as cross-entropy with sgd
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(hidden_layer_sizes=(20,), max_iter=600, verbose=10, 
               solver='sgd', learning_rate='constant', learning_rate_init=0.001)
    elif config['model_type'] == 'kNN':
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(n_neighbors=1, metric='cosine')

def extract_audioset_features(paths, path2gt): 
    """Extracts Audioset features and their corresponding ground_truth and identifiers (the path).

       Audioset features are extracted from non-overlapping audio patches of 0.96 seconds, 
       where each audio patch covers 64 mel bands and 96 frames of 10 ms each.

       We repeat ground_truth and identifiers to fit the number of extracted Audioset features.
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

    # 2) Load Tensorflow model to extract Audioset features
    with tf.Graph().as_default(), tf.Session() as sess:
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, 'vggish_model.ckpt')
        features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
        extracted_feat = sess.run([embedding_tensor], feed_dict={features_tensor: input_data})
        feature = np.squeeze(np.asarray(extracted_feat))

    return [feature, ground_truth, identifiers]


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
        first_batch = True
        pointer = -1 # it enables to access the remaining data (below) when our batch is too big for the available data
        for pointer in tqdm(range(len(paths_train)//config['train_batch'])):
            paths = paths_train[(pointer)*config['train_batch']:(pointer+1)*config['train_batch']]
            [x, y, refs] = extract_audioset_features(paths, path2gt_train)
            if first_batch:
                [X, Y, IDS] = [x, y, refs]
                first_batch = False
            else:
                 X = np.concatenate((X, x), axis=0)
                 Y = np.concatenate((Y, y), axis=0)
                 IDS = np.concatenate((IDS, refs), axis=0)

        # remaining train data
        paths = paths_train[(pointer+1)*config['train_batch']:]
        if not len(paths) == 0:
            [x, y, refs] = extract_audioset_features(paths, path2gt_train)
            if first_batch:
                [X, Y, IDS] = [x, y, refs]
                first_batch = False
            else:
                X = np.concatenate((X, x), axis=0)
                Y = np.concatenate((Y, y), axis=0)
                IDS = np.concatenate((IDS, refs), axis=0)

        # create a directory where to store the extracted training features
        audio_representations_folder = DATA_FOLDER + 'audio_representations/'
        if not os.path.exists(audio_representations_folder):
            os.makedirs(audio_representations_folder)
        training_data_file_name = 'training_data_' + str(config['dataset']) + '_' +  str(random.randint(0,10000))
        np.savez(audio_representations_folder + training_data_file_name, X=X, Y=Y, IDS=IDS)

    print(X.shape)
    print(Y.shape)
    print(Y)

    print('Fitting model..')
    model = define_classification_model()
    model.fit(X, Y)

    print('Evaluating model..')
    pred = []
    identifiers = []
    first_batch = True
    pointer = -1 # it enables to access the remaining data (below) when our batch is too big for the available data
    for pointer in tqdm(range(len(paths_test)//config['test_batch'])):
        paths = paths_test[(pointer)*config['test_batch']:(pointer+1)*config['test_batch']]
        [x, _, refs] = extract_audioset_features(paths, path2gt_test)
        if first_batch:
            [pred, identifiers] = [model.predict(x), refs]
            first_batch = False
        else:
            pred = np.concatenate((pred, model.predict(x)), axis=0)
            identifiers = np.concatenate((identifiers, refs), axis=0)
      
    # remaining test data
    paths = paths_test[(pointer+1)*config['test_batch']:]
    if not len(paths) == 0:
        [x, _, refs] = extract_audioset_features(paths, path2gt_test)
        if first_batch:
            [pred, identifiers] = [model.predict(x), refs]
            first_batch = False
        else:
            pred = np.concatenate((pred, model.predict(x)), axis=0)
            identifiers = np.concatenate((identifiers, refs), axis=0)

    # agreggating same ID: majority voting
    y_pred = []
    y_true = []
    for pt in paths_test:
        y_pred.append(np.argmax(np.bincount(pred[np.where(identifiers==pt)]))) # majority voting
        y_true.append(int(path2gt_test[pt]))

    # print and store the results
    conf_matrix = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    experiments_folder = DATA_FOLDER + 'experiments/'
    if not os.path.exists(experiments_folder):
        os.makedirs(experiments_folder)
    results_file_name = 'results_' + str(config['dataset']) + '_' +  str(random.randint(0,10000)) + '.txt'
    to = open(experiments_folder + results_file_name, 'w')
    to.write(str(config) + '\n')
    to.write(str(conf_matrix) + '\n')
    to.write('Accuracy: ' + str(acc))
    to.close()
    print(config)
    print(conf_matrix)    
    print('Accuracy: ' + str(acc))

