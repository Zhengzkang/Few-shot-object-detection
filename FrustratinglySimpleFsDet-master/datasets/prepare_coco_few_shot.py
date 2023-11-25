import argparse
import json
import os
import random
import time

import sys
sys.path.append('..')  # TODO: ugly but works for now
print("Path: {}".format(sys.path))
from class_splits import CLASS_SPLITS
from fsdet.config import get_cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["coco", "isaid"], required=True,
                        help="Dataset name")
    parser.add_argument("--class-split", type=str, required=True, dest="class_split",
                        help="Split of classes into base classes and novel classes")
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 3, 5, 10, 30],
                        help="Amount of annotations per class for fine tuning")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 9],
                        help="Range of seeds to run. Just a single seed or two seeds representing a range with 2nd "
                             "argument being inclusive as well!")
    parser.add_argument("--no-shuffle", action="store_false", default=True, dest="shuffle",
                        help="Shuffle images prior to sampling of annotations.")
    args = parser.parse_args()
    return args


def get_data_path():  # get path to training data annotations
    # probably use cfg.DATA_DIR[args.dataset] if necessary
    if args.dataset == "coco":
        return os.path.join(cfg.ROOT_DIR, cfg.TRAIN_ANNOS['coco'])
    elif args.dataset == "isaid":
        return os.path.join(cfg.ROOT_DIR, cfg.TRAIN_ANNOS['isaid'])
    else:
        raise ValueError("Dataset {} is not supported!".format(args.dataset))


def generate_seeds(args):
    start = time.time()
    data_path = get_data_path()
    data = json.load(open(data_path))

    new_all_cats = []  # category "objects"
    for cat in data['categories']:
        new_all_cats.append(cat)

    id2img = {}
    for i in data['images']:
        id2img[i['id']] = i
    # same but shorter: id2img = {i['id']: i for i in data['images']}

    # tuples of category names
    # TODO: base- and novel classes do not matter when sampling few-shot data, but may be important when saving them!
    base_classes = tuple(CLASS_SPLITS[args.dataset][args.class_split]['base'])
    novel_classes = tuple(CLASS_SPLITS[args.dataset][args.class_split]['novel'])
    all_classes = tuple(base_classes + novel_classes)

    coco_cat_id_to_name = {c['id']: c['name'] for c in new_all_cats}
    # Need make sure, 'all_classes' are all contained in 'coco_cat_id_to_name'
    assert len(all_classes) <= len(coco_cat_id_to_name) \
           and len(set(all_classes + tuple(coco_cat_id_to_name.values()))) == len(coco_cat_id_to_name), \
           "Error, inconsistency with categories defined in the dataset and in the class split: {} and {}".\
           format(coco_cat_id_to_name.values(), all_classes)

    cat_name_to_annos = {i: [] for i in all_classes}
    for anno in data['annotations']:
        if anno['iscrowd'] == 1:
            continue
        cat_name = coco_cat_id_to_name[anno['category_id']]
        if cat_name not in cat_name_to_annos:  # if base and novel classes do not sum up to all classes in the dataset
            continue
        else:
            cat_name_to_annos[cat_name].append(anno)

    if len(args.seeds) == 1:
        seeds = [args.seeds[0]]
    else:
        assert len(args.seeds) == 2
        seeds = range(args.seeds[0], args.seeds[1] + 1)
    for i in seeds:
        print("Generating seed {}".format(i))
        for cat_name in all_classes:
            print("Generating data for class {}".format(cat_name))
            img_id_to_annos = {}
            for anno in cat_name_to_annos[cat_name]:
                if anno['image_id'] in img_id_to_annos:
                    img_id_to_annos[anno['image_id']].append(anno)
                else:
                    img_id_to_annos[anno['image_id']] = [anno]

            for shots in args.shots:
                sample_annos = []  # annotations
                sample_imgs = []  # images
                sample_img_ids = []  # ids of sampled images, just used for duplicate checks
                if cat_name in base_classes:
                    assert cat_name not in novel_classes
                    if cfg.BASE_SHOT_MULTIPLIER == -1:
                        target_shots = len(cat_name_to_annos[cat_name])  # should be all available annos
                        print("Using all available {} annotations for base class {}!"
                              .format(target_shots, cat_name))
                    else:
                        assert cfg.BASE_SHOT_MULTIPLIER > 0
                        target_shots = cfg.BASE_SHOT_MULTIPLIER * shots
                        print("Generating {}x{} shot data for base class {}"
                              .format(cfg.BASE_SHOT_MULTIPLIER, shots, cat_name))
                else:
                    assert cat_name in novel_classes
                    target_shots = shots
                    print("Generating {} shot data for novel class {}"
                          .format(shots, cat_name))
                img_ids = list(img_id_to_annos.keys())
                # while True:
                    # img_ids = random.sample(list(img_id_to_annos.keys()), shots)
                # TODO: probably use random.sample(img_ids, 1) in a 'while True'-loop?
                if args.shuffle:
                    shuffle_seed = i  # Same order for same seeds, but should not matter...
                    random.seed(shuffle_seed)
                    print("shuffling images")
                    random.shuffle(img_ids)
                else:
                    print("not shuffling images prior to sampling!")
                for img_id in img_ids:
                    if img_id in sample_img_ids:  # only necessary if we iterate multiple times through all images
                        continue
                    if len(img_id_to_annos[img_id]) + len(sample_annos) > target_shots:
                        # TODO: This condition may lead to following:
                        #  1. For k=5 shots and if each image had exactly 2 annotations per class we finally only
                        #  have four annotations for that class -> probably too few annotations
                        #  2. In contrast to other approaches, they allow for taking multiple annotations from the
                        #  same image (even more: they only want ALL annotations from an image (for a certain class)
                        #  or none at all) (as support data) -> unknown consequences
                        continue
                    sample_annos.extend(img_id_to_annos[img_id])  # add all annotations of image with id 'img_id' with class 'c'
                    sample_imgs.append(id2img[img_id])  # add the image with id 'img_id'
                    sample_img_ids.append(img_id)
                    assert len(sample_imgs) <= len(sample_annos), \
                        "Error, got {} images but only {} annotations!".format(len(sample_imgs), len(sample_annos))
                    if len(sample_annos) == target_shots:
                        break
                # TODO: Probably convert assertion to a warning.
                assert len(sample_annos) == target_shots, "Wanted {} shots, but only found {} annotations!"\
                    .format(target_shots, len(sample_annos))
                new_data = data.copy()
                new_data['images'] = sample_imgs
                new_data['annotations'] = sample_annos
                new_data['categories'] = new_all_cats
                # Note: even if we sample more annotations for base classes we use the original 'shots' in the file
                # name for clarity!
                save_path = get_save_path_seeds(data_path, cat_name, shots, i)
                with open(save_path, 'w') as f:
                    # json.dump(new_data, f)
                    json.dump(new_data, f, indent=2)  # Easier to check files manually
    end = time.time()
    m, s = divmod(int(end-start), 60)
    print("Created few-shot data for {} shots and {} seeds in {}m {}s"
          .format(len(args.shots), len(seeds), m, s))


def get_save_path_seeds(path, cls, shots, seed):
    s = path.split('/')
    train_name = cfg.TRAIN_SPLIT[args.dataset]
    prefix = 'full_box_{}shot_{}_{}'.format(shots, cls, train_name)
    save_dir = os.path.join(cfg.DATA_SAVE_PATH_PATTERN[args.dataset].format(args.class_split), 'seed{}'.format(seed))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, prefix + '.json')
    return save_path


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    cfg = get_cfg()  # get default config to obtain the correct load- and save paths for the created data
    generate_seeds(args)
