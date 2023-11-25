import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import torch
from collections import OrderedDict
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.structures import BoxMode
from detectron2.utils.logger import create_small_table

from fsdet.evaluation.evaluator import DatasetEvaluator
from class_splits import CLASS_SPLITS, get_ids_from_names


class COCOEvaluator(DatasetEvaluator):
    """
    Evaluate instance detection outputs using COCO's metrics and APIs.
    """

    def __init__(self, dataset_name, cfg, distributed, output_dir=None, dataset='coco', file_suffix=""):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:
                    "json_file": the path to the COCO format annotation
                Or it must be in detectron2's standard dataset format
                    so it can be converted to COCO format automatically.
            cfg (CfgNode): config instance
            distributed (True):
                if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump results.
        """
        self._distributed = distributed
        self._output_dir = output_dir
        os.makedirs(self._output_dir, exist_ok=True)  # just to be sure
        self._dataset_name = dataset_name
        # save file names
        # raw detections passed to this class
        self._raw_predictions_file = os.path.join(self._output_dir, "instances_predictions{}.pth".format(file_suffix))
        # coco detections used to build coco_eval object
        self._coco_detections_file = os.path.join(self._output_dir, "coco_instances_results{}.json".format(file_suffix))
        self._summary_file = os.path.join(self._output_dir, "summary_results{}.txt".format(file_suffix))
        open(self._summary_file, 'w').close()  # create empty file, or delete content is existent

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            self._logger.warning(
                f"json_file was not found in MetaDataCatalog for '{dataset_name}'")

            cache_path = convert_to_coco_json(dataset_name, output_dir)
            self._metadata.json_file = cache_path
        # TODO: problematic when using class_split 'none_all'
        self._is_splits = "all" in dataset_name or "base" in dataset_name \
            or "novel" in dataset_name
        self._class_split = self._metadata.class_split
        # Note: we use 'thing_ids' over 'all_ids' because 'meta_coco' will override 'thing_ids' appropriately
        self._all_class_ids = self._metadata.thing_ids
        self._base_class_ids = self._metadata.get("base_ids", None)
        self._novel_class_ids = self._metadata.get("novel_ids",  None)
        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset

    def reset(self):
        self._predictions = []
        self._coco_results = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            # TODO this is ugly
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(
                    instances, input["image_id"])
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))

            if not comm.is_main_process():
                return {}

        if len(self._predictions) == 0:
            self._logger.warning(
                "[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            with PathManager.open(self._raw_predictions_file, "wb") as f:
                torch.save(self._predictions, f)

        self._results = OrderedDict()
        if "instances" in self._predictions[0]:
            self._eval_predictions()
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self):
        """
        Evaluate self._predictions on the instance detection task.
        Fill self._results with the metrics of the instance detection task.
        """
        self._logger.info("Preparing results for COCO format ...")
        self._coco_results = list(
            itertools.chain(*[x["instances"] for x in self._predictions]))

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in self._coco_results:
                result["category_id"] = reverse_id_mapping[result["category_id"]]

        if self._output_dir:
            self._logger.info("Saving results to {}".format(self._coco_detections_file))
            with PathManager.open(self._coco_detections_file, "w") as f:
                f.write(json.dumps(self._coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        if self._is_splits:
            self._results["bbox"] = {}
            for split, classes, names in [
                    ("all", self._all_class_ids, self._metadata.get("thing_classes")),
                    ("base", self._base_class_ids, self._metadata.get("base_classes")),
                    ("novel", self._novel_class_ids, self._metadata.get("novel_classes"))]:
                if "all" not in self._dataset_name and \
                        split not in self._dataset_name:
                    continue
                coco_eval = (
                    _evaluate_predictions_on_coco(
                        self._coco_api, self._coco_results, "bbox", classes,
                    )
                    if len(self._coco_results) > 0
                    else None  # cocoapi does not handle empty results very well
                )
                res_ = self._derive_coco_results(
                    coco_eval, "bbox", class_names=names, split=split
                )
                res = {}
                for metric in res_.keys():
                    if len(metric) <= 4:
                        if split == "all":
                            res[metric] = res_[metric]
                        elif split == "base":
                            res["b"+metric] = res_[metric]
                        elif split == "novel":
                            res["n"+metric] = res_[metric]
                self._results["bbox"].update(res)

            # add "AP" if not already in
            if "AP" not in self._results["bbox"]:
                if "nAP" in self._results["bbox"]:
                    self._results["bbox"]["AP"] = self._results["bbox"]["nAP"]
                else:
                    self._results["bbox"]["AP"] = self._results["bbox"]["bAP"]
        else:
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api, self._coco_results, "bbox",
                )
                if len(self._coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )
            res = self._derive_coco_results(
                coco_eval, "bbox",
                class_names=self._metadata.get("thing_classes")
            )
            self._results["bbox"] = res

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None, split=''):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """
        def log_info_and_append(file, str):
            with open(file, 'a') as f:
                f.write(str + 2*'\n')
            self._logger.info(str)

        def get_per_category_ap_table(coco_eval, class_names, iou_low=0.5, iou_high=0.95):
            # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
            def _get_thr_ind(coco_eval, thr):
                ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                               (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
                iou_thr = coco_eval.params.iouThrs[ind]
                assert np.isclose(iou_thr, thr)
                return ind
            precisions = coco_eval.eval["precision"]
            # precision has dims (iou, recall, cls, area range, max dets)
            assert len(class_names) == precisions.shape[2], "{},{}".format(len(class_names), precisions.shape[2])
            ind_lo = _get_thr_ind(coco_eval, iou_low)
            ind_hi = _get_thr_ind(coco_eval, iou_high)
            results_per_category = []
            for idx, name in enumerate(class_names):
                # area range index 0: all area ranges
                # max dets index -1: typically 100 per image
                precision = precisions[ind_lo:(ind_hi + 1), :, idx, 0, -1]
                precision = precision[precision > -1]
                ap = np.mean(precision) if precision.size else float("nan")
                results_per_category.append(("{}".format(name), float(ap * 100)))
            # tabulate it
            N_COLS = min(6, len(results_per_category) * 2)
            results_flatten = list(itertools.chain(*results_per_category))
            results_2d = itertools.zip_longest(
                *[results_flatten[i::N_COLS] for i in range(N_COLS)])
            if iou_low == 0.5 and iou_high == 0.95:
                ap_name = "AP"
            elif iou_low == iou_high:
                ap_name = "AP{}".format(round(iou_low * 100))  # e.g. AP50
            else:  # very rare case
                ap_name = "AP{}-{}".format(round(iou_low * 100), round(iou_high * 100))  # e.g. AP65-80
            table = tabulate(
                results_2d,
                tablefmt="pipe",
                floatfmt=".3f",
                headers=["category", ap_name] * (N_COLS // 2),
                numalign="left",
            )
            return results_per_category, table

        metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl"]
        split_str = '({} classes)'.format(split) if split != '' else split
        if coco_eval is None:
            self._logger.warn("No predictions from the model! Set scores to -1")
            return {metric: -1 for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100) \
                for idx, metric in enumerate(metrics)
        }
        tmp_str = "Evaluation results for {} {}: \n".format(iou_type, split_str) + create_small_table(results)
        log_info_and_append(self._summary_file, tmp_str)

        if not class_names:
            return results

        # get per-class AP@0.5:0.95, log result, append to file and update 'results'
        results_per_category, table = get_per_category_ap_table(coco_eval, class_names, iou_low=0.5, iou_high=0.95)
        tmp_str = "Per-category {} AP {}: \n".format(iou_type, split_str) + table
        log_info_and_append(self._summary_file, tmp_str)
        results.update({"AP-" + name: ap for name, ap in results_per_category})

        # get per-class AP@0.5, log result, append to file and update 'results'
        # TODO: probably just do this if 'class_names' are all class names
        #  -> What's with the case of base training? We then wouldn't have any per-class AP@50 results
        results_per_category, table = get_per_category_ap_table(coco_eval, class_names, iou_low=0.5, iou_high=0.5)
        tmp_str = "Per-category {} AP50 {}: \n".format(iou_type, split_str) + table
        log_info_and_append(self._summary_file, tmp_str)
        results.update({"AP@0.5-" + name: ap for name, ap in results_per_category})  # TODO: necessary for AP@0.5?

        return results


def instances_to_coco_json(instances, img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        results.append(result)
    return results


def _evaluate_predictions_on_coco(coco_gt, coco_results, iou_type, catIds=None):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    if catIds is not None:
        coco_eval.params.catIds = catIds
    coco_eval.evaluate()
    coco_eval.accumulate()
    # TODO: probably redirect stdout to self._summary_file, in order to also get the output of coco_eval.summarize()
    #  into the text file?
    coco_eval.summarize()

    return coco_eval
