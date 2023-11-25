import os
from fsdet.config.config import get_cfg


def main():
    cfg = get_cfg()
    dataset = "coco"  # coco, isaid
    coco_class_splits = ["voc_nonvoc"]  # voc_nonvoc, none_all
    isaid_class_splits = ["vehicle_nonvehicle"]  # vehicle_nonvehicle, none_all, experiment1, experiment2, experiment3
    gpu_ids = [0]
    num_threads = 2
    phase = 2  # phase 1: base-training, phase 2: fine-tuning
    #bs = 16
    shots = [10]  # shots to evaluate on
    seeds = [0]  # seeds to evaluate on
    eval_mode = 'single'  # all, single, last
    # normally, 10k steps. Note: it automatically subtracts 1 to fit the odd iteration in the checkpoint file names
    iterations = [10000]
    layers = 50  # 50, 101
    classifier = 'fc'  # fc, cosine
    tfa = False  # False: randinit surgery
    unfreeze = False  # False: freeze feature extractor while fine-tuning
    # Modify test config options (e.g. for quick test hyperparameter tuning).
    #  Note: these configs are not saved into a config file, the change is just temporary for this certain run!
    if dataset == "coco":
        class_splits = coco_class_splits
        opts = [
            'MODEL.ROI_HEADS.SCORE_THRESH_TEST', 0.05,
            'TEST.DETECTIONS_PER_IMAGE', 100,
            'MODEL.RPN.PRE_NMS_TOPK_TEST', 1000,
            'MODEL.RPN.POST_NMS_TOPK_TEST', 1000
        ]
    elif dataset == "isaid":
        class_splits = isaid_class_splits
        opts = [
            'MODEL.ROI_HEADS.SCORE_THRESH_TEST', 0.01,
            'TEST.DETECTIONS_PER_IMAGE', 300,
            'MODEL.RPN.PRE_NMS_TOPK_TEST', 2000,
            'MODEL.RPN.POST_NMS_TOPK_TEST', 1500
        ]
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))
    if eval_mode != 'single':  # to prevent multiple execution of inference on all or the last checkpoint!
        iterations = [-1]
    if phase == 1:
        mode = 'base'
        pattern = 'faster_rcnn_R_{}_FPN_{}.yaml'
        for class_split in class_splits:
            for iteration in iterations:
                config_file = os.path.join(
                    cfg.CONFIG_DIR_PATTERN[dataset].format(class_split),
                    pattern.format(layers, mode)
                )
                run_inference(gpu_ids, num_threads, config_file, eval_mode, iteration, opts)
    else:
        assert phase == 2
        assert len(shots) > 0 and len(seeds) > 0
        mode = 'all'
        pattern = 'faster_rcnn_R_{}_FPN_ft{}_{}{}{}{}{}.yaml'
        classifier_str = '_{}'.format(classifier)
        unfreeze_str = '_unfreeze' if unfreeze else ''
        tfa_str = '_TFA' if tfa else ''
        for class_split in class_splits:
            for shot in shots:
                for seed in seeds:
                    for iteration in iterations:
                        # TODO: Possible problems in future:
                        #  1. We hard-code mode 'all'
                        #  2. We don't use a suffix (as in 'run_experiments.py')
                        config_file = os.path.join(
                            cfg.CONFIG_DIR_PATTERN[dataset].format(class_split),
                            'seed{}'.format(seed),
                            'ft_only_novel' if mode == 'novel' else 'ft' + classifier_str + unfreeze_str,  # sub dir
                            pattern.format(layers, classifier_str, mode, '_{}shot'.format(shot), unfreeze_str, tfa_str, '')
                        )
                        run_inference(gpu_ids, num_threads, config_file, eval_mode, iteration, opts)


def run_inference(gpu_ids, num_threads, config_file, eval_mode, iteration, opts):
    assert eval_mode in ['all', 'single', 'last']
    if eval_mode == 'single':  # certain iteration
        eval_mode_str = "--eval-only --eval-iter {}".format(iteration)
    elif eval_mode == 'all':  # all available iterations
        eval_mode_str = "--eval-all"
    else:  # only last iteration
        eval_mode_str = "--eval-only"
    opts_str = '' if not opts else '--opts ' + separate(opts, ' ')
    base_cmd = "python3 -m tools.test_net"
    cmd = "OMP_NUM_THREADS={} CUDA_VISIBLE_DEVICES={} {} --config-file {} --num-gpus {} {} {}"\
        .format(num_threads, separate(gpu_ids, ','), base_cmd, config_file, len(gpu_ids), eval_mode_str, opts_str)
    os.system(cmd)


# note: separate(elements, ' ') == *elements
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
