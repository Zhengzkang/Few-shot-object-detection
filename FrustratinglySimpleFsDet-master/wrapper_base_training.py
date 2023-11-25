import os


def main():
    dataset = "coco"  # coco, isaid
    coco_class_split = "voc_nonvoc"  # voc_nonvoc, none_all
    isaid_class_split = "vehicle_nonvehicle"  # vehicle_nonvehicle, none_all, experiment1, experiment2, experiment3
    gpu_ids = [0]
    num_threads = 2  # two threads seem to be a bit faster than just one, but four threads are as fast as two threads!
    bs = 16
    lr = 0.02  # 0.02 for bs=16. Set to -1 for automatic linear scaling!
    layers = 50  # 50, 101
    override_config = True
    if dataset == "coco":
        class_split = coco_class_split
    elif dataset == "isaid":
        class_split = isaid_class_split
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))
    run_base_training(dataset, class_split, gpu_ids, num_threads, layers, bs, lr, override_config)


def run_base_training(dataset, class_split, gpu_ids, num_threads, layers, bs, lr=-1.0, override_config=False):
    base_cmd = "python3 -m tools.run_base_training"
    override_config_str = ' --override-config' if override_config else ''
    cmd = "{} --dataset {} --class-split {} --gpu-ids {} --num-threads {} --layers {} --bs {} --lr {}{}"\
        .format(base_cmd, dataset, class_split, separate(gpu_ids, ' '), num_threads, layers, bs, lr, override_config_str)
    os.system(cmd)


def separate(elements, separator):
    res = ''
    if not isinstance(elements, (list, tuple)):
        return str(elements)
    assert len(elements) > 0, "need at least one element in the collection {}!".format(elements)
    if len(elements) == 1:
        return str(elements[0])
    for element in elements:
        res += '{}{}'.format(str(element), separator)
    return res[:-1]  # remove trailing separator


if __name__ == '__main__':
    main()
