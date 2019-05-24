import os
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.io import wavfile


def load_path2gt(paths_file, config):
    """ Given the path, construct the ground truth vectors.
        This function heavily relies on path2gt_datasets(.),
        where the relation between the path and ground truth
        are defined.
    """
    paths = list()
    path2gt = dict()
    path2onehot = dict() # REMOVE IF NOT USED!
    pf = open(paths_file)
    for path in pf.readlines():
        path = path.rstrip('\n')
        paths.append(path)
        label = path2gt_datasets(path, config['dataset'])
        path2gt[path] = label
        path2onehot[path] = label2onehot(label, config['num_classes_dataset'])
    return paths, path2gt, path2onehot


def label2onehot(label, num_classes):
    """ Convert class label to one hot vector.
        Example: label2onehot(label=2, num_classes=5) > array([0., 0., 1., 0., 0.])
    """
    onehot = np.zeros(num_classes)
    onehot[label] = 1
    return onehot


def path2gt_datasets(path, dataset):
    """ Given the audio path, it returns the ground truth label.
        Define HERE a new dataset to employ this code with other data.
    """
    if dataset == 'GTZAN':
        if 'blues' in path:
            return 0
        elif 'classical' in path:
            return 1
        elif 'country' in path:
            return 2
        elif 'disco' in path:
            return 3
        elif 'hiphop' in path:
            return 4
        elif 'jazz' in path:
            return 5
        elif 'metal' in path:
            return 6
        elif 'pop' in path:
            return 7
        elif 'reggae' in path:
            return 8
        elif 'rock' in path:
            return 9
        else:
            print('Did not find the corresponding ground truth (' + str(path) + ')!')

    else:
            print('Did not find the implementation of ' + str(dataset) + ' dataset!')


def matrix_visualization(matrix,title=None):
    """ Visualize 2D matrices like spectrograms or feature maps.
    """
    plt.figure()
    plt.imshow(np.flipud(matrix.T),interpolation=None)
    plt.colorbar()
    if title!=None:
        plt.title(title)
    plt.show()


def wavefile_to_waveform(wav_file, features_type):
    data, samplerate = sf.read(wav_file)
    if features_type == 'vggish':
        tmp_name = str(int(np.random.rand(1)*1000000)) + '.wav'
        sf.write(tmp_name, data, samplerate, subtype='PCM_16')
        sr, wav_data = wavfile.read(tmp_name)
        os.remove(tmp_name)
        # sr, wav_data = wavfile.read(wav_file) # as done in VGGish Audioset
        assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
        data = wav_data / 32768.0  # Convert to [-1.0, +1.0]
  
    # at least one second of samples, if not repead-pad
    src_repeat = data
    while (src_repeat.shape[0] < sr): 
        src_repeat = np.concatenate((src_repeat, data), axis=0)
        data = src_repeat[:sr]

    return data, sr

