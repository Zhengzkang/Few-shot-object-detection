# Original coco annotations. COCO category IDs to COCO category names
COCO_CATS_ID_TO_NAME = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}
# Inverse dict of COCO category names to COCO category IDs
COCO_CATS_NAME_TO_ID = {v: k for k, v in COCO_CATS_ID_TO_NAME.items()}

# TODO: Category IDs of iSAID original train and val annotations differ!!!
# As we use iSAID_patches, and because category IDs of patches train and val annotations are the same, we use the IDs
# specified in the patches annotation files!
ISAID_CATS_ID_TO_NAME = {
    1: 'Small_Vehicle',
    2: 'Large_Vehicle',
    3: 'plane',
    4: 'storage_tank',
    5: 'ship',
    6: 'Swimming_pool',
    7: 'Harbor',
    8: 'tennis_court',
    9: 'Ground_Track_Field',
    10: 'Soccer_ball_field',
    11: 'baseball_diamond',
    12: 'Bridge',
    13: 'basketball_court',
    14: 'Roundabout',
    15: 'Helicopter'
}
# Inverse dict of iSAID category names to iSAID category IDs
ISAID_CATS_NAME_TO_ID = {v: k for k, v in ISAID_CATS_ID_TO_NAME.items()}

# Note: use lists for generating a collection implicitly! With tuples we could later run into problems while
# iterating over them!
#####################
# COCO class splits #
#####################
_COCO_VOC_NAMES = ['airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table',
                   'dog', 'horse', 'motorcycle', 'person', 'potted plant', 'sheep', 'couch', 'train', 'tv']
_COCO_NONVOC_NAMES = [i for i in COCO_CATS_ID_TO_NAME.values() if i not in _COCO_VOC_NAMES]

# Mainly for debugging and creation of coco-base trained models used for other datasets for better initialization
# (and for having a good initialization for FPN!)
_COCO_NO_NAMES = []
_COCO_ALL_NAMES = [i for i in COCO_CATS_ID_TO_NAME.values() if i not in _COCO_NO_NAMES]

######################
# iSAID class splits #
######################
# Experimental class split
_ISAID_VEHICLE_NAMES = ['Small_Vehicle', 'Large_Vehicle', 'plane', 'ship', 'Helicopter']
_ISAID_NONVEHICLE_NAMES = [i for i in ISAID_CATS_ID_TO_NAME.values() if i not in _ISAID_VEHICLE_NAMES]

# For debugging the program
_ISAID_NO_NAMES = []
_ISAID_ALL_NAMES = [i for i in ISAID_CATS_ID_TO_NAME.values() if i not in _ISAID_NO_NAMES]

# Experiment 1
# (novel: all vehicle classes but small vehicle, because it's a very difficult category with much training data.
# ship is comparably difficult so we can compare performance of difficult base class ship and novel class small vehicle)
_ISAID_EXPERIMENT1_NOVEL = ['Helicopter', 'ship', 'plane', 'Large_Vehicle']
_ISAID_EXPERIMENT1_BASE = [i for i in ISAID_CATS_ID_TO_NAME.values() if i not in _ISAID_EXPERIMENT1_NOVEL]

# Experiment 2 (some easy novel categories which should be easily adapted to)
_ISAID_EXPERIMENT2_NOVEL = ['baseball_diamond', 'Soccer_ball_field', 'Roundabout']
_ISAID_EXPERIMENT2_BASE = [i for i in ISAID_CATS_ID_TO_NAME.values() if i not in _ISAID_EXPERIMENT2_NOVEL]

# Experiment 3
# (rare classes to see if a split of abundant and rare classes could be better than a 'normal' training on all classes)
_ISAID_EXPERIMENT3_NOVEL = ['Ground_Track_Field', 'Helicopter', 'Roundabout',
                            'Soccer_ball_field', 'basketball_court', 'baseball_diamond']  # order is from rare to common
_ISAID_EXPERIMENT3_BASE = [i for i in ISAID_CATS_ID_TO_NAME.values() if i not in _ISAID_EXPERIMENT3_NOVEL]

_ISAID_EXPERIMENT3_BEST1_NOVEL = ['Roundabout']
_ISAID_EXPERIMENT3_BASEBALL_NOVEL = ['baseball_diamond']
_ISAID_EXPERIMENT3_HELICOPTER_NOVEL = ['Helicopter']
_ISAID_EXPERIMENT3_GTF_NOVEL = ['Ground_Track_Field']
_ISAID_EXPERIMENT3_SOCCER_NOVEL = ['Soccer_ball_field']
_ISAID_EXPERIMENT3_WORST1_NOVEL = ['basketball_court']

_ISAID_EXPERIMENT3_BEST3_NOVEL = ['Roundabout', 'baseball_diamond', 'Helicopter']
_ISAID_EXPERIMENT3_WORST3_NOVEL = ['basketball_court', 'Soccer_ball_field', 'Ground_Track_Field']

_SAID_EXPERIMENT3_COMMON3_BASE = ['Small_Vehicle', 'Large_Vehicle', 'ship']
_SAID_EXPERIMENT3_COMMON6_BASE = ['Small_Vehicle', 'Large_Vehicle', 'ship', 'plane', 'storage_tank', 'Harbor']

_ISAID_SIMILAR1_BASE = ['tennis_court']
_ISAID_SIMILAR1_NOVEL = ['basketball_court']

_ISAID_DIFFERENT1_BASE = ['plane']
_ISAID_DIFFERENT1_NOVEL = ['basketball_court']

_ISAID_DIFFERENT2_BASE = ['Harbor']
_ISAID_DIFFERENT2_NOVEL = ['basketball_court']

_ISAID_STORAGE_ROUNDABOUT_BASE = ['storage_tank']
_ISAID_STORAGE_ROUNDABOUT_NOVEL = ['Roundabout']

_ISAID_PLANE_ROUNDABOUT_BASE = ['plane']
_ISAID_PLANE_ROUNDABOUT_NOVEL = ['Roundabout']

_ISAID_LV_ROUNDABOUT_BASE = ['Large_Vehicle']
_ISAID_LV_ROUNDABOUT_NOVEL = ['Roundabout']

ALL_CLASSES = {
    "coco": _COCO_ALL_NAMES,
    "isaid": _ISAID_ALL_NAMES
}

# Note: We mainly use category names because both, indices and specific category IDs can be ambiguous and may not be
# defined the same way, everywhere they are used. Sometimes, categories are stored inside dictionaries, which do not
# preserve order, instead of lists that do preserve order, what makes the use of indices hard. Some IDs are not
# contiguous (as in MS COCO), what makes the use of indices hard. Sometimes, specific category IDs differ (as in
# iSAID train and val), therefore we prefer to only use category names.
CLASS_SPLITS = {}

# TODO: probably adjust class split names (sometimes, we have two inverse class splits. Their names wouldn't be much
#  distinguishable (e.g. X_Y and Y_X)

# COCO has 80 classes
CLASS_SPLITS["coco"] = {
    "voc_nonvoc": {
        "base": _COCO_NONVOC_NAMES,
        "novel": _COCO_VOC_NAMES
    },
    "none_all": {
        "base": _COCO_ALL_NAMES,
        "novel": _COCO_NO_NAMES
    }
}

# iSAID has 15 classes
CLASS_SPLITS["isaid"] = {
    "vehicle_nonvehicle": {
        "base": _ISAID_NONVEHICLE_NAMES,
        "novel": _ISAID_VEHICLE_NAMES
    },
    "none_all": {
        "base": _ISAID_ALL_NAMES,
        "novel": _ISAID_NO_NAMES
    },
    "experiment1": {
        "base": _ISAID_EXPERIMENT1_BASE,
        "novel": _ISAID_EXPERIMENT1_NOVEL
    },
    "experiment2": {
        "base": _ISAID_EXPERIMENT2_BASE,
        "novel": _ISAID_EXPERIMENT2_NOVEL
    },
    "experiment3": {
        "base": _ISAID_EXPERIMENT3_BASE,
        "novel": _ISAID_EXPERIMENT3_NOVEL
    },
    "experiment3_best1": {  # TODO: change to "*best1_novel"?
        "base": _ISAID_EXPERIMENT3_BASE,
        "novel": _ISAID_EXPERIMENT3_BEST1_NOVEL
    },
    "experiment3_best3": {  # TODO: change to "*best3_novel"?
        "base": _ISAID_EXPERIMENT3_BASE,
        "novel": _ISAID_EXPERIMENT3_BEST3_NOVEL
    },
    "experiment3_worst1": {  # TODO: change to "*worst1_novel"?
        "base": _ISAID_EXPERIMENT3_BASE,
        "novel": _ISAID_EXPERIMENT3_WORST1_NOVEL
    },
    "experiment3_worst3": {  # TODO: change to "*worst3_novel"?
        "base": _ISAID_EXPERIMENT3_BASE,
        "novel": _ISAID_EXPERIMENT3_WORST3_NOVEL
    },
    "experiment3_common3": {  # TODO: change to "*common3_base"?
        "base": _SAID_EXPERIMENT3_COMMON3_BASE,
        "novel": _ISAID_EXPERIMENT3_NOVEL
    },
    "experiment3_common6": {  # TODO: change to "*common6_base"?
        "base": _SAID_EXPERIMENT3_COMMON6_BASE,
        "novel": _ISAID_EXPERIMENT3_NOVEL
    },
    "similar1": {
        "base": _ISAID_SIMILAR1_BASE,
        "novel": _ISAID_SIMILAR1_NOVEL
    },
    "different1": {
        "base": _ISAID_DIFFERENT1_BASE,
        "novel": _ISAID_DIFFERENT1_NOVEL
    },
    "different2": {
        "base": _ISAID_DIFFERENT2_BASE,
        "novel": _ISAID_DIFFERENT2_NOVEL
    },
    "storage_roundabout": {
        "base": _ISAID_STORAGE_ROUNDABOUT_BASE,
        "novel": _ISAID_STORAGE_ROUNDABOUT_NOVEL
    },
    "plane_roundabout": {
        "base": _ISAID_PLANE_ROUNDABOUT_BASE,
        "novel": _ISAID_PLANE_ROUNDABOUT_NOVEL
    },
    "lv_roundabout": {
        "base": _ISAID_LV_ROUNDABOUT_BASE,
        "novel": _ISAID_LV_ROUNDABOUT_NOVEL
    },
    "experiment3_baseball_novel": {
        "base": _ISAID_EXPERIMENT3_BASE,
        "novel": _ISAID_EXPERIMENT3_BASEBALL_NOVEL
    },
    "experiment3_helicopter_novel": {
        "base": _ISAID_EXPERIMENT3_BASE,
        "novel": _ISAID_EXPERIMENT3_HELICOPTER_NOVEL
    },
    "experiment3_gtf_novel": {
        "base": _ISAID_EXPERIMENT3_BASE,
        "novel": _ISAID_EXPERIMENT3_GTF_NOVEL
    },
    "experiment3_soccer_novel": {
        "base": _ISAID_EXPERIMENT3_BASE,
        "novel": _ISAID_EXPERIMENT3_SOCCER_NOVEL
    },
}

# xVIEW has 60 classes
CLASS_SPLITS["xview"] = {

}

# VAID contains 7 vehicle classes
CLASS_SPLITS["vaid"] = {

}


def get_ids_from_names(dataset, class_names):
    if dataset == 'coco':
        return [COCO_CATS_NAME_TO_ID[c] for c in class_names]
    elif dataset == 'isaid':
        return [ISAID_CATS_NAME_TO_ID[c] for c in class_names]
    else:
        raise ValueError("Error, no mapping available for dataset {}!".format(dataset))


def get_names_from_ids(dataset, ids):
    if dataset == 'coco':
        return [COCO_CATS_ID_TO_NAME[i] for i in ids]
    elif dataset == 'isaid':
        return [ISAID_CATS_ID_TO_NAME[i] for i in ids]
    else:
        raise ValueError("Error, no mapping available for dataset {}!".format(dataset))


def check_splits():
    print("checking all class splits for consistency...")
    for (dataset, num_cls) in [("coco", 80), ("xview", 60), ("isaid", 15), ("vaid", 7)]:
        for class_split in CLASS_SPLITS[dataset].values():
            assert "novel" in class_split.keys() and "base" in class_split.keys()
            novel_classes = class_split["novel"]
            base_classes = class_split["base"]
            assert len(novel_classes) + len(base_classes) == num_cls, \
                "Error in class splits length {} and {}".format(novel_classes, base_classes)
            assert len(set(novel_classes + base_classes)) == num_cls, \
                "Error, found duplicates in class splits {} and {}".format(novel_classes, base_classes)
    print("... no errors found!")


if __name__ == '__main__':
    check_splits()
