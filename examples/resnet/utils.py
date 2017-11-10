from __future__ import division, print_function

cifar_sizes = [8, 20, 32, 44, 56, 110]
i1k_sizes = [18, 34, 50, 101, 152]


# Takes network name and size and returns a dictionary of required parameters
def get_network_params(dataset, size, batch_size):
    nw_params = dict()
    if dataset == 'cifar10':
        nw_params['metric_names'] = ['Cross_Ent_Loss', 'Misclass']
        nw_params['en_top5'] = False
        dataset_sizes = cifar_sizes
        if size in dataset_sizes:
            # Num of resnet modules required for cifar10
            nw_params['num_resnet_mods'] = (size - 2) // 6
            # Change iter_interval to print every epoch
            nw_params['iter_interval'] = 50000 // batch_size
            nw_params['learning_schedule'] = [84 * nw_params['iter_interval'],
                                              124 * nw_params['iter_interval']]
            print("Learning Schedule: " + str(nw_params['learning_schedule']))
            # CIFAR10 doesn't use bottleneck
            nw_params['en_bottleneck'] = False
            # There are 10 classes so setting length of label axis to 10
            nw_params['num_classes'] = 10
        else:
            raise ValueError("Invalid CIFAR10 size. Select from " + str(dataset_sizes))
        return nw_params
    elif dataset == 'cifar100':
        nw_params['metric_names'] = ['Cross_Ent_Loss', 'Misclass']
        nw_params['en_top5'] = False
        dataset_sizes = cifar_sizes
        if size in dataset_sizes:
            # Num of resnet modules required for cifar10
            nw_params['num_resnet_mods'] = (size - 2) // 6
            # Change iter_interval to print every epoch
            nw_params['iter_interval'] = 50000 // batch_size
            nw_params['learning_schedule'] = [84 * nw_params['iter_interval'],
                                              124 * nw_params['iter_interval']]
            print("Learning Schedule: " + str(nw_params['learning_schedule']))
            # CIFAR10 doesn't use bottleneck
            nw_params['en_bottleneck'] = False
            # There are 10 classes so setting length of label axis to 10
            nw_params['num_classes'] = 100
        else:
            raise ValueError("Invalid CIFAR10 size. Select from " + str(dataset_sizes))
        return nw_params
    elif dataset == 'i1k':
        nw_params['metric_names'] = ['Cross_Ent_Loss', 'Top5_Err', 'Top1_Err']
        nw_params['en_top5'] = True
        dataset_sizes = i1k_sizes
        if size in dataset_sizes:
            # Enable or disable bottleneck depending on resnet size
            if(size in [18, 34]):
                nw_params['en_bottleneck'] = False
            else:
                nw_params['en_bottleneck'] = True
            # Change iter_interval to print every epoch.
            nw_params['iter_interval'] = 1301000 // batch_size
            nw_params['learning_schedule'] = [30 * nw_params['iter_interval'],
                                              60 * nw_params['iter_interval']]
            print("Learning Schedule: " + str(nw_params['learning_schedule']))
            nw_params['num_resnet_mods'] = 0
            # of Classes
            nw_params['num_classes'] = 1000
        else:
            raise ValueError("Invalid i1k size. Select from " + str(dataset_sizes))
        return nw_params
    elif dataset == 'i1k100':
        nw_params['metric_names'] = ['Cross_Ent_Loss', 'Top5_Err', 'Top1_Err']
        nw_params['en_top5'] = True
        dataset_sizes = i1k_sizes
        if size in dataset_sizes:
            # Enable or disable bottleneck depending on resnet size
            if(size in [18, 34]):
                nw_params['en_bottleneck'] = False
            else:
                nw_params['en_bottleneck'] = True
            # Change iter_interval to print every epoch.
            nw_params['iter_interval'] = 126101 // batch_size
            nw_params['learning_schedule'] = [30 * nw_params['iter_interval'],
                                              60 * nw_params['iter_interval']]
            print("Learning Schedule: " + str(nw_params['learning_schedule']))
            nw_params['num_resnet_mods'] = 0
            # of Classes
            nw_params['num_classes'] = 100
        else:
            raise ValueError("Invalid i1k size. Select from " + str(dataset_sizes))
        return nw_params
    else:
        raise NameError("Invalid Dataset. Dataset should be either cifar10 or i1k")


# Sets learning rate based on the current iteration
def set_lr(base_lr, step, learning_schedule, gamma):
    lr = base_lr
    if((step >= learning_schedule[0]) and (step < learning_schedule[1])):
        lr = base_lr * gamma
    if(step >= learning_schedule[1]):
        lr = base_lr * gamma * gamma
    return lr
