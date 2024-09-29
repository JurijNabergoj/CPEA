import os
import pprint
import torch
import numpy as np

hierarchy_mapping = {
    # Train Split
    'n00000001': ('fish', 'aquarium_fish'),
    'n00000032': ('fish', 'flatfish'),
    'n00000067': ('fish', 'ray'),
    'n00000073': ('fish', 'shark'),
    'n00000091': ('fish', 'trout'),

    'n00000054': ('flowers', 'orchids'),
    'n00000062': ('flowers', 'poppies'),
    'n00000070': ('flowers', 'roses'),
    'n00000082': ('flowers', 'sunflowers'),
    'n00000092': ('flowers', 'tulips'),

    'n00000009': ('food_containers', 'bottles'),
    'n00000010': ('food_containers', 'bowls'),
    'n00000016': ('food_containers', 'cans'),
    'n00000028': ('food_containers', 'cups'),
    'n00000061': ('food_containers', 'plates'),

    'n00000000': ('fruit_and_vegetables', 'apples'),
    'n00000051': ('fruit_and_vegetables', 'mushrooms'),
    'n00000053': ('fruit_and_vegetables', 'oranges'),
    'n00000057': ('fruit_and_vegetables', 'pears'),
    'n00000083': ('fruit_and_vegetables', 'sweet_peppers'),

    'n00000022': ('household_electrical_devices', 'clock'),
    'n00000039': ('household_electrical_devices', 'computer_keyboard'),
    'n00000040': ('household_electrical_devices', 'lamp'),
    'n00000086': ('household_electrical_devices', 'telephone'),
    'n00000087': ('household_electrical_devices', 'television'),

    'n00000005': ('household_furniture', 'bed'),
    'n00000020': ('household_furniture', 'chair'),
    'n00000025': ('household_furniture', 'couch'),
    'n00000084': ('household_furniture', 'table'),
    'n00000094': ('household_furniture', 'wardrobe'),

    'n00000012': ('large_man-made_outdoor_things', 'bridge'),
    'n00000017': ('large_man-made_outdoor_things', 'castle'),
    'n00000037': ('large_man-made_outdoor_things', 'house'),
    'n00000068': ('large_man-made_outdoor_things', 'road'),
    'n00000076': ('large_man-made_outdoor_things', 'skyscraper'),

    'n00000023': ('large_natural_outdoor_scenes', 'cloud'),
    'n00000033': ('large_natural_outdoor_scenes', 'forest'),
    'n00000049': ('large_natural_outdoor_scenes', 'mountain'),
    'n00000060': ('large_natural_outdoor_scenes', 'plain'),
    'n00000071': ('large_natural_outdoor_scenes', 'sea'),

    'n00000027': ('reptiles', 'crocodiles'),
    'n00000029': ('reptiles', 'dinosaurs'),
    'n00000044': ('reptiles', 'lizards'),
    'n00000078': ('reptiles', 'snakes'),
    'n00000093': ('reptiles', 'turtles'),

    'n00000047': ('trees', 'maple_tree'),
    'n00000052': ('trees', 'oak_tree'),
    'n00000056': ('trees', 'palm_tree'),
    'n00000059': ('trees', 'pine_tree'),
    'n00000096': ('trees', 'willow_tree'),

    'n00000008': ('vehicles_1', 'bicycle'),
    'n00000013': ('vehicles_1', 'bus'),
    'n00000048': ('vehicles_1', 'motorcycle'),
    'n00000058': ('vehicles_1', 'pickup_truck'),

    'n00000041': ('vehicles_2', 'lawn_mower'),
    'n00000069': ('vehicles_2', 'rocket'),
    'n00000081': ('vehicles_2', 'tram'),
    'n00000085': ('vehicles_2', 'tank'),
    'n00000089': ('vehicles_2', 'tractor'),
    'n00000090': ('vehicles_2', 'train'),

    # Validation Split
    'n00000003': ('large_carnivores', 'bear'),
    'n00000042': ('large_carnivores', 'leopard'),
    'n00000043': ('large_carnivores', 'lion'),
    'n00000088': ('large_carnivores', 'tiger'),
    'n00000097': ('large_carnivores', 'wolf'),

    'n00000015': ('large_omnivores_and_herbivores', 'camel'),
    'n00000019': ('large_omnivores_and_herbivores', 'cattle'),
    'n00000021': ('large_omnivores_and_herbivores', 'chimpanzee'),
    'n00000031': ('large_omnivores_and_herbivores', 'elephant'),
    'n00000038': ('large_omnivores_and_herbivores', 'kangaroo'),

    'n00000026': ('non-insect_invertebrates', 'crab'),
    'n00000045': ('non-insect_invertebrates', 'lobster'),
    'n00000077': ('non-insect_invertebrates', 'snail'),
    'n00000079': ('non-insect_invertebrates', 'spider'),
    'n00000099': ('non-insect_invertebrates', 'worm'),

    'n00000036': ('small_mammals', 'hamster'),
    'n00000050': ('small_mammals', 'mouse'),
    'n00000065': ('small_mammals', 'rabbit'),
    'n00000074': ('small_mammals', 'shrew'),
    'n00000080': ('small_mammals', 'squirrel'),

    # Test Split
    'n00000004': ('aquatic_mammals', 'beaver'),
    'n00000030': ('aquatic_mammals', 'dolphin'),
    'n00000055': ('aquatic_mammals', 'otter'),
    'n00000072': ('aquatic_mammals', 'seal'),
    'n00000095': ('aquatic_mammals', 'whale'),

    'n00000006': ('insects', 'bee'),
    'n00000007': ('insects', 'beetle'),
    'n00000014': ('insects', 'butterfly'),
    'n00000018': ('insects', 'caterpillar'),
    'n00000024': ('insects', 'cockroach'),

    'n00000034': ('medium_mammals', 'fox'),
    'n00000063': ('medium_mammals', 'porcupine'),
    'n00000064': ('medium_mammals', 'possum'),
    'n00000066': ('medium_mammals', 'raccoon'),
    'n00000075': ('medium_mammals', 'skunk'),

    'n00000002': ('people', 'baby'),
    'n00000011': ('people', 'boy'),
    'n00000035': ('people', 'girl'),
    'n00000046': ('people', 'man'),
    'n00000098': ('people', 'woman'),
}

level2_mapping = {
    # Train Split
    'fish': 0,
    'flowers': 1,
    'food_containers': 2,
    'fruit_and_vegetables': 3,
    'household_electrical_devices': 4,
    'household_furniture': 5,
    'large_man-made_outdoor_things': 6,
    'large_natural_outdoor_scenes': 7,
    'reptiles': 8,
    'trees': 9,
    'vehicles_1': 10,
    'vehicles_2': 11,

    # Validation Split
    'large_carnivores': 12,
    'large_omnivores_and_herbivores': 13,
    'non-insect_invertebrates': 14,
    'small_mammals': 15,

    # Test Split
    'aquatic_mammals': 16,
    'insects': 17,
    'medium_mammals': 18,
    'people': 19,
}

class_to_subclass_dict = {
    0: [1, 32, 67, 73, 91],
    1: [54, 62, 70, 82, 92],
    2: [9, 10, 16, 28, 61],
    3: [0, 51, 53, 57, 83],
    4: [22, 39, 40, 86, 87],
    5: [5, 20, 25, 84, 94],
    6: [12, 17, 37, 68, 76],
    7: [23, 33, 49, 60, 71],
    8: [27, 29, 44, 78, 93],
    9: [47, 52, 56, 59, 96],
    10: [8, 13, 48, 58],
    11: [41, 69, 81, 85, 89, 90],
    12: [3, 42, 43, 88, 97],
    13: [15, 19, 21, 31, 38],
    14: [26, 45, 77, 79, 99],
    15: [36, 50, 65, 74, 80],
    16: [4, 30, 55, 72, 95],
    17: [6, 7, 14, 18, 24],
    18: [34, 63, 64, 66, 75],
    19: [2, 11, 35, 46, 98],
}



def ensure_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    # print(pred.size())
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm


# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules import Module


class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''

    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def prototypical_loss(input, target, n_support):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))

    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    # FIXME when torch will support where as np
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

    query_samples = input.to('cpu')[query_idxs]
    dists = euclidean_dist(query_samples, prototypes)

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()

    return loss_val, acc_val
