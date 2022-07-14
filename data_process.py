import numpy as np 
import utils

def open_closed_noisy_labels(dataset1, dataset1_label, dataset2, closed_noise_type='symmetric', openset_noise_rate=0.2, closed_set_noise_rate=0.2, num_classes=10, random_seed=1):
    # dataset1 is corrupted by dataset2 
    # dataset1 and dataset2 .npy format
    # not -> dataset1 and dataset2 do not have same classes, e.g., CIFAR-10 and SVHN (MNIST, *CIFAR-100)

    num_total_1, num_total_2 = int(dataset1.shape[0]), int(dataset2.shape[0])

    noise_rate = float(openset_noise_rate + closed_set_noise_rate)
    num_noisy_labels_1 = int(noise_rate * num_total_1)
    num_open_noisy_labels_1, num_closed_noisy_labels_1 = int(openset_noise_rate * num_total_1), int(closed_set_noise_rate * num_total_1)

    np.random.seed(random_seed)
    corrupted_labels_index_1, corrupted_labels_index_2 = np.random.choice(num_total_1, num_noisy_labels_1, replace=False), np.random.choice(num_total_2, num_open_noisy_labels_1, replace=False)
    corrupted_open_noisy_labels_index_1, corrupted_closed_noisy_labels_index_1  = corrupted_labels_index_1[:num_open_noisy_labels_1], corrupted_labels_index_1[num_open_noisy_labels_1:-1]
    print(corrupted_open_noisy_labels_index_1)
    print(corrupted_closed_noisy_labels_index_1)

    # open_set_corruption (images corruption)
    dataset1[corrupted_open_noisy_labels_index_1] = dataset2[corrupted_labels_index_2]

    # closed_set_corruption (labels corruption)
    labels = dataset1_label[corrupted_closed_noisy_labels_index_1]
    labels = labels[:, np.newaxis]
    if closed_noise_type == 'symmetric':
        noisy_labels, _, _ = utils.noisify_multiclass_symmetric(labels, closed_set_noise_rate, random_state=random_seed, nb_classes=num_classes)
        dataset1_label[corrupted_closed_noisy_labels_index_1] = noisy_labels.squeeze()
    elif closed_noise_type == 'pairflip':
        noisy_labels, _, _ = utils.noisify_pairflip(labels, closed_set_noise_rate, random_state=random_seed, nb_classes=num_classes)
        dataset1_label[corrupted_closed_noisy_labels_index_1] = noisy_labels.squeeze()
    return dataset1, dataset1_label




