import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer


def _load_label_names():
    """
    Load the label names from file
    """
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

"""
aquatic mammals    beaver, dolphin, otter, seal, whale
fish    aquarium fish, flatfish, ray, shark, trout
flowers    orchids, poppies, roses, sunflowers, tulips
food containers    bottles, bowls, cans, cups, plates
fruit and vegetables    apples, mushrooms, oranges, pears, sweet peppers
household electrical devices    clock, computer keyboard, lamp, telephone, television
household furniture    bed, chair, couch, table, wardrobe
insects    bee, beetle, butterfly, caterpillar, cockroach
large carnivores    bear, leopard, lion, tiger, wolf
large man-made outdoor things    bridge, castle, house, road, skyscraper
large natural outdoor scenes    cloud, forest, mountain, plain, sea
large omnivores and herbivores    camel, cattle, chimpanzee, elephant, kangaroo
medium-sized mammals    fox, porcupine, possum, raccoon, skunk
non-insect invertebrates    crab, lobster, snail, spider, worm
people    baby, boy, girl, man, woman
reptiles    crocodile, dinosaur, lizard, snake, turtle
small mammals    hamster, mouse, rabbit, shrew, squirrel
trees    maple, oak, palm, pine, willow
vehicles 1    bicycle, bus, motorcycle, pickup truck, train
vehicles 2    lawn-mower, rocket, streetcar, tank, tractor
"""



def load_cfar100_batch(cifar100_dataset_folder_path):
    """
    Load a batch of the dataset
    """
    with open(cifar100_dataset_folder_path + '/train', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')
    with open(cifar100_dataset_folder_path + '/meta', mode='rb') as file:
        meta = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    fine_labels_id = batch['fine_labels']
    coarse_labels_id = batch['coarse_labels']
    fine_labels = meta['fine_label_names']
    coarse_labels = meta['coarse_label_names']

    return features, fine_labels, coarse_labels, fine_labels_id, coarse_labels_id


def display_stats(cifar100_dataset_folder_path,  sample_id):
    """
    Display Stats of the the dataset
    """
    features, fine_labels, coarse_labels, fine_labels_id, coarse_labels_id  = \
        load_cfar100_batch(cifar100_dataset_folder_path)

    if not (0 <= sample_id < len(features)):
        print('{} samples.  {} is out of range.'.format(len(features), sample_id))
        return None

    #print('Samples: {}'.format(len(features)))
    #print('Fine Label Counts: {}'.format(dict(zip(*np.unique(fine_labels, return_counts=True)))))
    #print('Coarse Label Counts: {}'.format(dict(zip(*np.unique(coarse_labels, return_counts=True)))))
    #print('First 20 fine labels: {}'.format(fine_labels[:20]))
    #print('First 20 coarse labels: {}'.format(coarse_labels[:20]))


    sample_image = features[sample_id]
    sample_fine_label = fine_labels_id[sample_id]
    sample_coarse_label = coarse_labels_id[sample_id]
    sample_fine_label_name = fine_labels[sample_fine_label]
    sample_coarse_label_name = coarse_labels[sample_coarse_label]


    print('\nExample of Image {}:'.format(sample_id))
    print('Image - Min Value: {} Max Value: {}'.format(sample_image.min(), sample_image.max()))
    print('Image - Shape: {}'.format(sample_image.shape))
    print('Coarse Label - Label Id: {} Name: {}'.format(sample_fine_label,sample_coarse_label_name))
    print('Fine Label - Label Id: {} Name: {}'.format(sample_coarse_label,sample_fine_label_name))
    plt.axis('off')
    plt.imshow(sample_image)


def _preprocess_and_save(normalize, one_hot_encode, features, labels, filename):
    """
    Preprocess data and save it to file
    """
    features = normalize(features)
    labels = one_hot_encode(labels)

    pickle.dump((features, labels), open(filename, 'wb'))


def preprocess_and_save_data(cifar100_dataset_folder_path, normalize, one_hot_encode):
    """
    Preprocess Training and Validation Data
    """
    n_batches = 1
    valid_features = []
    valid_labels = []

    for batch_i in range(1, n_batches + 1):
        features, labels = load_cfar100_batch(cifar100_dataset_folder_path, batch_i)
        validation_count = int(len(features) * 0.1)

        # Prprocess and save a batch of training data
        _preprocess_and_save(
            normalize,
            one_hot_encode,
            features[:-validation_count],
            labels[:-validation_count],
            'preprocess_batch_' + str(batch_i) + '.p')

        # Use a portion of training batch for validation
        valid_features.extend(features[-validation_count:])
        valid_labels.extend(labels[-validation_count:])

    # Preprocess and Save all validation data
    _preprocess_and_save(
        normalize,
        one_hot_encode,
        np.array(valid_features),
        np.array(valid_labels),
        'preprocess_validation.p')

    with open(cifar100_dataset_folder_path + '/test_batch', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    # load the test data
    test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = batch['labels']

    # Preprocess and Save all test data
    _preprocess_and_save(
        normalize,
        one_hot_encode,
        np.array(test_features),
        np.array(test_labels),
        'preprocess_test.p')


def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]


def load_preprocess_training_batch(batch_id, batch_size):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    filename = 'preprocess_batch_' + str(batch_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))

    # Return the training data in batches of size <batch_size> or less
    return batch_features_labels(features, labels, batch_size)


def display_image_predictions(features, labels, predictions):
    n_classes = 10
    label_names = _load_label_names()
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(range(n_classes))
    label_ids = label_binarizer.inverse_transform(np.array(labels))

    fig, axies = plt.subplots(nrows=4, ncols=2)
    fig.tight_layout()
    fig.suptitle('Softmax Predictions', fontsize=20, y=1.1)

    n_predictions = 3
    margin = 0.05
    ind = np.arange(n_predictions)
    width = (1. - 2. * margin) / n_predictions

    for image_i, (feature, label_id, pred_indicies, pred_values) in enumerate(zip(features, label_ids, predictions.indices, predictions.values)):
        pred_names = [label_names[pred_i] for pred_i in pred_indicies]
        correct_name = label_names[label_id]

        axies[image_i][0].imshow(feature)
        axies[image_i][0].set_title(correct_name)
        axies[image_i][0].set_axis_off()

        axies[image_i][1].barh(ind + margin, pred_values[::-1], width)
        axies[image_i][1].set_yticks(ind + margin)
        axies[image_i][1].set_yticklabels(pred_names[::-1])
        axies[image_i][1].set_xticks([0, 0.5, 1.0])
