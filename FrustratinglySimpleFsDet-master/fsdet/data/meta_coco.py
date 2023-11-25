import numpy as np
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO

import contextlib
import io
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

from class_splits import CLASS_SPLITS
from fsdet.config import get_cfg
cfg = get_cfg()
COCOLIKE_METADATA_NAMES = {}

"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""

__all__ = ["register_meta_cocolike", "get_cocolike_metadata_names"]


# Cache the metadata names since they do not change at runtime.
def get_cocolike_metadata_names(dataset='coco'):
    if dataset not in COCOLIKE_METADATA_NAMES:
        COCOLIKE_METADATA_NAMES[dataset] = _get_cocolike_metadata_names(dataset=dataset)
    return COCOLIKE_METADATA_NAMES[dataset]


def _get_cocolike_metadata_names(dataset='coco'):
    train_name = cfg.TRAIN_SPLIT[dataset]
    test_name = cfg.TEST_SPLIT[dataset]
    assert train_name not in test_name
    assert test_name not in train_name
    ret = {}
    # register whole training and testing datasets. Whole test dataset is used for fine-tuning inference
    #  TODO: what is the complete training dataset used for?
    for split in [train_name, test_name]:
        args = (dataset, split)
        ret['{}_{}_all'.format(*args)] = args
    # register class-split dependent datasets (just needed to get the correct metadata!):
    #  base training datasets
    #  testing datasets for base training, for fine tuning just the novel detector and for testing of
    #   a complete fine-tuning. Last testing set is required to have class splits because this allows for
    #   having different colors for different class splits
    for class_split in CLASS_SPLITS[dataset].keys():
        args = (dataset, class_split, train_name, 'base')
        ret['{}_{}_{}_{}'.format(*args)] = args
        for prefix in ['base', 'novel', 'all']:
            args = (dataset, class_split, test_name, prefix)
            ret['{}_{}_{}_{}'.format(*args)] = args
    # register training datasets for fine tuning the whole detector
    for class_split in CLASS_SPLITS[dataset].keys():
        for prefix in ['all', 'novel']:
            for shot in cfg.VALID_FEW_SHOTS:
                for seed in range(cfg.MAX_SEED_VALUE + 1):  # maximum seed value is inclusive!
                    args = (dataset, class_split, train_name, prefix, shot, seed)
                    ret['{}_{}_{}_{}_{}shot_seed{}'.format(*args)] = args
    return ret


def load_cocolike_json(dataset, json_file, image_root, metadata, dataset_name):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection.
    Args:
        dataset(str): dataset identifier
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str): the directory where the images in this json file exists.
        metadata: meta data associated with dataset_name
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    train_name = cfg.TRAIN_SPLIT[dataset]
    test_name = cfg.TEST_SPLIT[dataset]
    cocolike_metadata_names = get_cocolike_metadata_names(dataset)
    assert dataset_name in cocolike_metadata_names
    dataset_labels = cocolike_metadata_names[dataset_name]
    id_map = metadata["thing_dataset_id_to_contiguous_id"]
    dataset_dicts = []
    ann_keys = ["iscrowd", "bbox", "category_id"]
    if len(dataset_labels) == 2 or len(dataset_labels) == 4:
        # Note: we don't care about the actual labels. We load the whole dataset and all annotations nonetheless.
        #  The labels are just important to obtain the correct metadata.
        json_file = PathManager.get_local_path(json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            coco_api = COCO(json_file)
        # sort indices for reproducible results
        img_ids = sorted(list(coco_api.imgs.keys()))
        imgs = coco_api.loadImgs(img_ids)
        anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
        imgs_anns = list(zip(imgs, anns))
        for (img_dict, anno_dict_list) in imgs_anns:
            record = {}
            record["file_name"] = os.path.join(
                image_root, img_dict["file_name"]
            )
            record["height"] = img_dict["height"]
            record["width"] = img_dict["width"]
            image_id = record["image_id"] = img_dict["id"]

            objs = []
            for anno in anno_dict_list:
                assert anno["image_id"] == image_id
                assert anno.get("ignore", 0) == 0

                obj = {key: anno[key] for key in ann_keys if key in anno}

                obj["bbox_mode"] = BoxMode.XYWH_ABS
                if obj["category_id"] in id_map:
                    obj["category_id"] = id_map[obj["category_id"]]
                    objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    else:  # ~ 'is_shots'
        assert len(dataset_labels) == 6
        assert 'shot' in dataset_name  # normally not necessary, but resembles the old version of this code
        (_, class_split, _, prefix, shot, seed) = dataset_labels
        fileids = {}
        cls_ind_anno_count = {}  # class index to amount of annotations
        split_dir = cfg.DATA_SAVE_PATH_PATTERN[dataset].format(class_split)
        split_dir = os.path.join(split_dir, 'seed{}'.format(seed))
        for idx, cls in enumerate(metadata["thing_classes"]):
            json_file = os.path.join(split_dir, "full_box_{}shot_{}_{}.json".format(shot, cls, train_name))
            json_file = PathManager.get_local_path(json_file)
            with contextlib.redirect_stdout(io.StringIO()):
                coco_api = COCO(json_file)
            img_ids = sorted(list(coco_api.imgs.keys()))
            imgs = coco_api.loadImgs(img_ids)
            anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
            fileids[idx] = list(zip(imgs, anns))
            cls_ind_anno_count[idx] = sum(map(len, anns))
        ind_to_id = {v: k for k, v in id_map.items()}  # inverse map of index back to id
        base_id_to_ind = metadata["base_dataset_id_to_contiguous_id"]
        novel_id_to_ind = metadata["novel_dataset_id_to_contiguous_id"]
        max_annos = max(cls_ind_anno_count.values())
        assert cfg.FT_ANNOS_PER_IMAGE in ['one', 'all']
        for idx, fileids_ in fileids.items():
            dicts = []
            for (img_dict, anno_dict_list) in fileids_:
                if cfg.FT_ANNOS_PER_IMAGE == 'one':
                    for anno in anno_dict_list:
                        record = {}  # each record represents an image, not an annotation!
                        record["file_name"] = os.path.join(
                            image_root, img_dict["file_name"]
                        )
                        record["height"] = img_dict["height"]
                        record["width"] = img_dict["width"]
                        image_id = record["image_id"] = img_dict["id"]

                        assert anno["image_id"] == image_id
                        assert anno.get("ignore", 0) == 0

                        obj = {key: anno[key] for key in ann_keys if key in anno}

                        obj["bbox_mode"] = BoxMode.XYWH_ABS
                        obj["category_id"] = id_map[obj["category_id"]]
                        # for fine-tuning add each image multiple times with just one annotation each time
                        record["annotations"] = [obj]
                        dicts.append(record)
                else:
                    assert cfg.FT_ANNOS_PER_IMAGE == 'all'
                    record = {}
                    record["file_name"] = os.path.join(
                        image_root, img_dict["file_name"]
                    )
                    record["height"] = img_dict["height"]
                    record["width"] = img_dict["width"]
                    image_id = record["image_id"] = img_dict["id"]

                    objs = []
                    for anno in anno_dict_list:
                        assert anno["image_id"] == image_id
                        assert anno.get("ignore", 0) == 0

                        obj = {key: anno[key] for key in ann_keys if key in anno}

                        obj["bbox_mode"] = BoxMode.XYWH_ABS
                        # TODO: this is unnecessary for fine-tuning and probably unnecessary for
                        #  "only novel class"-trainings as well!
                        if obj["category_id"] in id_map:
                            obj["category_id"] = id_map[obj["category_id"]]
                            objs.append(obj)
                    record["annotations"] = objs
                    dicts.append(record)
            # Note: we cannot directly use {base|novel}_id_to_ind.values() because both use indices starting from
            #  zero. We have to transform the index (within all classes) back to the class-id and then may use this
            #  unique class-id to identify the class as either base class or novel class
            class_id = ind_to_id[idx]
            class_name = metadata["thing_classes"][idx]
            base_class_ids = base_id_to_ind.keys()
            novel_class_ids = novel_id_to_ind.keys()
            base_novel_str = ""
            if prefix == 'all':   # no need for adding more novel class annotations for only-novel fine-tuning
                if class_id in base_class_ids:  # we have a base class
                    assert class_id not in novel_class_ids
                    base_novel_str = "base "
                    if cfg.BASE_SHOT_MULTIPLIER > 0:
                        # Nothing to do, the 'base_shot_multiplier * shot' annotations have already been added to 'dicts'
                        target_shots = cfg.BASE_SHOT_MULTIPLIER * shot
                    elif cfg.BASE_SHOT_MULTIPLIER == -1 and cfg.NOVEL_OVERSAMPLING_FACTOR > 0:
                        # Nothing to do, use all annotations but do not balance anything!
                        target_shots = cls_ind_anno_count[idx]
                    else:
                        assert cfg.BASE_SHOT_MULTIPLIER == cfg.NOVEL_OVERSAMPLING_FACTOR == -1
                        dicts = duplicate_and_sample(dicts=dicts, target_size=max_annos)
                        target_shots = -1  # deactivate because balancing is not exact!
                else:  # we have a novel class
                    assert class_id in novel_class_ids
                    base_novel_str = "novel "
                    # over-sample the elements in the 'dicts'-list to ensure a more balanced training set for
                    #  fine-tuning, since we allow to sample more than K shots for base classes. This should work just
                    #  straightforward because the elements (images with annotations) in 'dataset_dicts' are used right
                    #  away, no post-processing is done. The elements are randomized and then batched. Therefore, it
                    #  does not matter if we have duplicates in this list!
                    if cfg.NOVEL_OVERSAMPLING_FACTOR > 0:
                        dicts = cfg.NOVEL_OVERSAMPLING_FACTOR * dicts  # Not the fastest solution, but single dicts are generally small
                        target_shots = cfg.NOVEL_OVERSAMPLING_FACTOR * shot
                    elif cfg.NOVEL_OVERSAMPLING_FACTOR == -1 and cfg.BASE_SHOT_MULTIPLIER > 0:
                        dicts = cfg.BASE_SHOT_MULTIPLIER * dicts
                        target_shots = cfg.BASE_SHOT_MULTIPLIER * shot
                    else:
                        assert cfg.NOVEL_OVERSAMPLING_FACTOR == cfg.BASE_SHOT_MULTIPLIER == -1
                        dicts = duplicate_and_sample(dicts=dicts, target_size=max_annos)
                        target_shots = -1  # deactivate because balancing is not exact!
            else:
                assert prefix == 'novel'  # prefix == 'base' and 'shot' in dataset_name is illegal!
                target_shots = shot
            if cfg.FT_ANNOS_PER_IMAGE == 'one' and len(dicts) > int(target_shots):
                print("Found {} annotations which is more than target annotations {}, "
                      "going to sample annotations randomly".format(len(dicts), int(target_shots)))
                dicts = np.random.choice(dicts, int(target_shots), replace=False)
            elif cfg.FT_ANNOS_PER_IMAGE == 'all' and target_shots > 0:
                assert sum(map(lambda record: len(record['annotations']), dicts)) <= int(target_shots)
            num_annos = sum(map(lambda record: len(record['annotations']), dicts))
            print("{}class: {}; ID: {}; available annotations {}; annotations used for training: {}"
                  .format(base_novel_str, class_name, class_id, cls_ind_anno_count[idx], num_annos))
            dataset_dicts.extend(dicts)
    return dataset_dicts


# Duplicate and sample given 'elements' to roughly fit the target size
def duplicate_and_sample(dicts, target_size):
    def dicts_len(dicts):
        return sum(map(lambda record: len(record['annotations']), dicts))
    target_dicts = []
    count = 0
    total_dict_annos = dicts_len(dicts)
    # duplicate all records as long as the whole list fits the target size
    factor = target_size // total_dict_annos
    target_dicts.extend(factor * dicts)
    count += factor * total_dict_annos
    assert count == dicts_len(target_dicts), "Inconsistency: {} and {}".format(count, dicts_len(target_dicts))
    # sample remaining annotations until we surpass target_size
    for record in dicts:  # TODO: random sample?
        assert count <= target_size
        anno_count = len(record['annotations'])
        if count + anno_count <= target_size:
            target_dicts.append(record)
            count += anno_count
        else:
            # we add another record if we are more below target_size than we were above if we would add another record
            # Note: this destroys our assertion above, so we immediately break
            if abs(count + anno_count - target_size) < abs(target_size - count):
                target_dicts.append(record)
                count += anno_count
            break

    assert count == dicts_len(target_dicts), "Inconsistency: {} and {}".format(count, dicts_len(target_dicts))
    return target_dicts


def register_meta_cocolike(dataset, name, metadata, imgdir, annofile):
    DatasetCatalog.register(
        name,
        lambda: load_cocolike_json(dataset, annofile, imgdir, metadata, name),
    )

    metadata_args = get_cocolike_metadata_names(dataset)[name]
    if len(metadata_args) == 4:
        # Either:
        #   - base training on only novel classes
        #   - inference on only base/novel classes or on all classes
        (_, _, _, prefix) = metadata_args
        metadata["thing_dataset_id_to_contiguous_id"] = metadata["{}_dataset_id_to_contiguous_id".format(prefix)]
        metadata["thing_classes"] = metadata["{}_classes".format(prefix)]
        metadata["thing_ids"] = metadata["{}_ids".format(prefix)]
    elif len(metadata_args) == 6:
        # Either:
        #  - fine-tuning on only novel classes
        #  - "regular" fine-tuning on all classes (base+novel)
        (_, _, _, prefix, _, _) = metadata_args
        metadata["thing_dataset_id_to_contiguous_id"] = metadata["{}_dataset_id_to_contiguous_id".format(prefix)]
        metadata["thing_classes"] = metadata["{}_classes".format(prefix)]
        metadata["thing_ids"] = metadata["{}_ids".format(prefix)]

    MetadataCatalog.get(name).set(
        json_file=annofile,
        image_root=imgdir,
        evaluator_type=dataset,
        dirname="datasets/{}".format(dataset),
        **metadata,
    )
