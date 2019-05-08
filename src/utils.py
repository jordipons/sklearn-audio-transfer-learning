import numpy as np
import matplotlib.pyplot as plt

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

