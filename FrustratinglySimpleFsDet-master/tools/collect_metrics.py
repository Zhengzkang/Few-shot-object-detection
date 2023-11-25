# TODO: probably rename to something like "collect Detectron2 style COCO metrics"

# TODO: make sure that this script works with all approaches absed on detectron!
#  -> it should especially work with 3rd approach. so look at its format and either adjust this script or adjust
#  evaluation of 3rd approach
import os.path
from statistics import mean, stdev
from fsdet.config import get_cfg
cfg = get_cfg()

mean_decimals = 1
std_decimals = 2

class_subsets = ['base', 'novel', 'all']
ious = ['0.50:0.95', '0.50', '0.75']
areas = ['small', 'medium', 'large', 'all']

dataset = 'isaid'  # isaid, coco
isaid_class_split = 'vehicle_nonvehicle'  # vehicle_nonvehicle, none_all, experiment{1|2|3}
coco_class_split = 'voc_nonvoc'  # voc_nonvoc, none_all
# general architecture
layers = 50  # 50, 101
# for base-training
base_iteration = 60000
# for fine-tuning
shots = 10  # 10, 50, 100
fine_iteration = 100000
seeds = [0, 1, 2]  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
tfa = False
classifier = 'fc'  # fc, cosine
unfreeze = False


def main():
    per_class_metrics, summary_metrics = get_statistics()
    metrics_to_print1 = [
        ['base', '0.50:0.95', 'all'],
        ['base', '0.50', 'all'],
        ['novel', '0.50:0.95', 'all'],
        ['novel', '0.50', 'all'],
        ['all', '0.50:0.95', 'all'],
        ['all', '0.50', 'all']
    ]
    metrics_to_print2 = [
        ['base', '0.50:0.95', 'all'],
        ['novel', '0.50:0.95', 'all'],
        ['all', '0.50:0.95', 'all'],
        ['base', '0.50', 'all'],
        ['novel', '0.50', 'all'],
        ['all', '0.50', 'all']
    ]
    _print_metrics(per_class_metrics, summary_metrics, summary_print_only=metrics_to_print1)


def default_summary_dict():
    return {subset: {iou: {area: [] for area in areas} for iou in ious} for subset in class_subsets}


def default_per_class_dict():
    return {iou: {} for iou in ious}


def _file_names(phase=2):
    if dataset == 'isaid':
        class_split = isaid_class_split
    elif dataset == 'coco':
        class_split = coco_class_split
    else:
        return
    metrics_file_pattern = 'summary_results_iter_{}.txt'
    if phase == 1:
        mode = 'base'  # Similar to 'run_base_training.py' and 'wrapper_inference.py' we hard-code the mode
        training_id = 'faster_rcnn_R_{}_FPN_{}'  # same as 'training_identifier' variable of 'run_base_training' script
        # build path, similar to 'ckpt_dir' variable in method 'get_cfg' of script 'run_base_training'
        yield os.path.join(
            cfg.CKPT_DIR_PATTERN[dataset].format(class_split),
            'faster_rcnn',
            training_id.format(layers, mode),
            'inference',
            metrics_file_pattern.format(base_iteration - 1)
        )
    else:
        assert phase == 2
        mode = 'all'  # # Similar to 'run_experiments.py' and 'wrapper_inference.py' we hard-code the mode
        # same as 'training_identifier' variable in 'run_experiments' script
        training_id = 'faster_rcnn_R_{}_FPN_ft{}_{}{}{}{}{}'
        classifier_str = '_{}'.format(classifier)
        unfreeze_str = '_unfreeze' if unfreeze else ''
        tfa_str = '_TFA' if tfa else ''
        shot_str = '_{}shot'.format(shots)
        for seed in seeds:
            # build path, similar to 'train_ckpt_save_dir' variable in 'get_cfg' method of script 'run_experiments.py'
            # Note: similar to 'wrapper_inference.py', we hard-code the suffix to be ''
            yield os.path.join(
                cfg.CKPT_DIR_PATTERN[dataset].format(class_split),
                'faster_rcnn',
                'seed{}'.format(seed),
                training_id.format(layers, classifier_str, mode, shot_str, unfreeze_str, tfa_str, ''),
                'inference',
                metrics_file_pattern.format(fine_iteration - 1)
            )


def get_statistics():
    per_class_metrics, summary_metrics = collect_metrics()
    for iou in ious:
        if not per_class_metrics[iou]:
            continue
        for cat_name, aps in per_class_metrics[iou].items():
            ap_mean = round(mean(aps), mean_decimals)
            ap_std = round(stdev(aps), std_decimals)
            per_class_metrics[iou][cat_name] = {
                'aps': aps,
                'mean': ap_mean,
                'std': ap_std
            }
    for class_subset in class_subsets:
        for iou in ious:
            for area in areas:
                maps = summary_metrics[class_subset][iou][area]
                if len(maps) == 0:  # no statistics for empty list!
                    continue
                map_mean = round(mean(maps), mean_decimals)
                map_std = round(stdev(maps), std_decimals)
                summary_metrics[class_subset][iou][area] = {
                    'maps': maps,
                    'mean': map_mean,
                    'std': map_std
                }
    return per_class_metrics, summary_metrics


def collect_metrics():
    debug = False
    if debug:
        filename = '../tmp_metrics.txt'
        return collect_metrics_single_file(filename)
    per_class_metrics = default_per_class_dict()
    summary_metrics = default_summary_dict()
    for file_name in _file_names(phase=2):
        with open(file_name, 'r') as f:
            _collect_metrics_single_file_recursive(open_file=f, current_line=None,
                                                   per_class_metrics=per_class_metrics, summary_metrics=summary_metrics)
    return per_class_metrics, summary_metrics


# mainly for debugging of parsing methods...
def collect_metrics_single_file(filename):
    per_class_metrics = default_per_class_dict()
    summary_metrics = default_summary_dict()

    with open(filename, 'r') as f:
        _collect_metrics_single_file_recursive(open_file=f, current_line=None,
                                               per_class_metrics=per_class_metrics, summary_metrics=summary_metrics)
    # _print_metrics(per_class_metrics, summary_metrics)
    return per_class_metrics, summary_metrics


def _collect_metrics_single_file_recursive(open_file, current_line, per_class_metrics, summary_metrics):
    if current_line is None:
        current_line = open_file.readline()
    while current_line:
        if current_line.startswith('Evaluation'):
            return _parse_summary_metrics(open_file, current_line, per_class_metrics, summary_metrics)
        if current_line.startswith('Per-category'):
            return _parse_per_class_metrics(open_file, current_line, per_class_metrics, summary_metrics)
        current_line = open_file.readline()


def _parse_summary_metrics(open_file, current_line, per_class_metrics, summary_metrics):
    def _return(line):
        return line.startswith('Evaluation') or line.startswith('Per-category')

    if 'all' in current_line:
        assert all(split not in current_line for split in ['base', 'novel'])
        split = 'all'
    elif 'base' in current_line:
        assert all(split not in current_line for split in ['all', 'novel'])
        split = 'base'
    else:
        assert 'novel' in current_line
        assert all(split not in current_line for split in ['all', 'base'])
        split = 'novel'
    assert current_line.startswith('Evaluation')  # shall be true for every call of this method
    # parse the line denoting the different summary metrics
    current_line = open_file.readline()  # move to next line, we now know, we're in a desired block
    assert not _return(current_line)
    tmp_line = current_line.strip()[1:-1]  # remove whitespace, as well as first and last char
    split_line = tmp_line.split('|')
    metric_names = [metric_name.strip() for metric_name in split_line]
    # parse/skip the separator between metric names and values
    current_line = open_file.readline()
    assert not _return(current_line)
    assert current_line.startswith('|:-')
    # parse metric values
    current_line = open_file.readline()
    assert not _return(current_line)
    tmp_line = current_line.strip()[1:-1]  # remove whitespace, as well as first and last char
    split_line = tmp_line.split('|')
    metric_values = [metric_value.strip() for metric_value in split_line]
    # zip metric names and values and update summary_metric dictionary
    assert len(metric_names) == len(metric_values)
    for name, value in zip(metric_names, metric_values):
        value = float(value)
        if name == 'AP':
            summary_metrics[split]['0.50:0.95']['all'].append(value)
        elif name == 'AP50':
            summary_metrics[split]['0.50']['all'].append(value)
        elif name == 'AP75':
            summary_metrics[split]['0.75']['all'].append(value)
        elif name == 'APs':
            summary_metrics[split]['0.50:0.95']['small'].append(value)
        elif name == 'APm':
            summary_metrics[split]['0.50:0.95']['medium'].append(value)
        elif name == 'APl':
            summary_metrics[split]['0.50:0.95']['large'].append(value)
        else:
            raise ValueError('Error at parsing, unknown metric name {}'.format(name))
    # move to next line until we reach a return condition
    while current_line:
        if _return(current_line):
            return _collect_metrics_single_file_recursive(open_file, current_line, per_class_metrics, summary_metrics)
        current_line = open_file.readline()


def _parse_per_class_metrics(open_file, current_line, per_class_metrics, summary_metrics):
    def _return(line):
        return line.startswith('Evaluation') or line.startswith('Per-category')

    if 'all' in current_line:
        assert all(split not in current_line for split in ['base', 'novel'])
        split = 'all'
    elif 'base' in current_line:
        assert all(split not in current_line for split in ['all', 'novel'])
        split = 'base'
    else:
        assert 'novel' in current_line
        assert all(split not in current_line for split in ['all', 'base'])
        split = 'novel'
    assert current_line.startswith('Per-category')  # shall be true for every call of this method
    if split != 'all':
        # per-category scores of base classes and novel classes are yet contained in the per-category scores for all
        #  classes!
        current_line = open_file.readline()  # move to next line, otherwise, we would directly come back to this method!
        return _collect_metrics_single_file_recursive(open_file, current_line, per_class_metrics, summary_metrics)
    # parse the iou used
    current_line = open_file.readline()
    assert not _return(current_line)
    tmp_line = current_line.strip()[1:-1]  # remove whitespace, as well as first and last char
    split_line = tmp_line.split('|')
    assert len(split_line) % 2 == 0
    metrics = [metric.strip() for metric in split_line[1::2]]
    assert len(set(metrics)) == 1
    metric = metrics[0]
    if metric == 'AP':
        iou = '0.50:0.95'
    elif metric == 'AP50':
        iou = '0.50'
    else:
        raise ValueError("Error at parsing, unknown metric {}".format(metric))
    # skip separator
    current_line = open_file.readline()
    assert not _return(current_line)
    assert current_line.startswith('|:-')
    # parse per-class metrics
    current_line = open_file.readline()
    while current_line and not _return(current_line):
        if not current_line.startswith('|') or current_line == '':
            break
        tmp_line = current_line.strip()[1:-1]  # remove whitespace, as well as first and last char
        split_line = tmp_line.split('|')
        categories = [cat_name.strip() for cat_name in split_line[0::2]]
        scores = [score.strip() for score in split_line[1::2]]
        assert len(categories) == len(scores)
        for cat, score in zip(categories, scores):
            assert (cat == '' and score == '') or (cat != '' and score != '')
            if cat == '':
                continue
            score = float(score)
            if cat not in per_class_metrics[iou]:
                per_class_metrics[iou][cat] = [score]
            else:
                per_class_metrics[iou][cat].append(score)
        current_line = open_file.readline()
    return _collect_metrics_single_file_recursive(open_file, current_line, per_class_metrics, summary_metrics)


def _print_metrics(per_category_metrics=None, summary_metrics=None, summary_print_only=None):
    if per_category_metrics is not None:
        for iou in ious:
            if not per_category_metrics[iou]:
                continue
            print("IoU={}".format(iou))
            for cat_name, ap in per_category_metrics[iou].items():
                print("{}: {}".format(cat_name, ap))
    if summary_metrics is not None:
        if summary_print_only is not None:  # restrict printed summary metrics
            for [class_subset, iou, area] in summary_print_only:
                maps = summary_metrics[class_subset][iou][area]
                print("{}: AP @[IoU = {} | area = {}] = {}".format(class_subset, iou, area, maps))
        else:
            for class_subset in class_subsets:
                print("~~~~ Summary {} metrics ~~~~".format(class_subset))
                for iou in ious:
                    for area in areas:
                        maps = summary_metrics[class_subset][iou][area]
                        if len(maps) == 0:
                            continue
                        print("AP @[IoU = {} | area = {}] = {}".format(iou, area, maps))


if __name__ == '__main__':
    main()
