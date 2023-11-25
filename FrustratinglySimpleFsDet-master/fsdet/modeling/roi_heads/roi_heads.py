"""Implement ROI_heads."""
import copy

import numpy as np
import torch
from torch import nn

import logging
from detectron2.data import MetadataCatalog
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone.resnet import BottleneckBlock, make_stage
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.sampling import subsample_labels
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from typing import Dict, List

from .box_head import build_box_head
from .fast_rcnn import ROI_HEADS_OUTPUT_REGISTRY, FastRCNNOutputLayers, FastRCNNOutputs

ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""

logger = logging.getLogger(__name__)


def build_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)


def select_foreground_proposals(proposals, bg_label):
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


class ROIHeads(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It contains logic of cropping the regions, extract per-region features,
    and make per-region predictions.

    It can have many variants, implemented as subclasses of this class.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(ROIHeads, self).__init__()

        # fmt: off
        self.batch_size_per_image     = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        self.test_score_thresh        = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.test_nms_thresh          = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_detections_per_img  = cfg.TEST.DETECTIONS_PER_IMAGE
        self.in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes              = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.proposal_append_gt       = cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT
        self.feature_strides          = {k: v.stride for k, v in input_shape.items()}
        self.feature_channels         = {k: v.channels for k, v in input_shape.items()}
        self.cls_agnostic_bbox_reg    = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.smooth_l1_beta           = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA
        # fmt: on

        # Matcher to assign box proposals to gt boxes
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )

        # Box2BoxTransform for bounding box regression
        self.box2box_transform = Box2BoxTransform(
            weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS
        )

    def _sample_proposals(self, matched_idxs, matched_labels, gt_classes):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes,
            self.batch_size_per_image,
            self.positive_sample_fraction,
            self.num_classes,
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns `self.batch_size_per_image` random samples from proposals and groundtruth boxes,
        with a fraction of positives that is no larger than `self.positive_sample_fraction.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:
                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                   then the ground-truth box is random)
                Other fields such as "gt_classes" that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(
                match_quality_matrix
            )
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # will filter the proposals again (by foreground/background,
                # etc), so we essentially index the data twice.
                for (
                    trg_name,
                    trg_value,
                ) in targets_per_image.get_fields().items():
                    if trg_name.startswith(
                        "gt_"
                    ) and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(
                            trg_name, trg_value[sampled_targets]
                        )
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros(
                        (len(sampled_idxs), 4)
                    )
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append(
                (gt_classes == self.num_classes).sum().item()
            )
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def forward(self, images, features, proposals, targets=None):
        """
        Args:
            images (ImageList):
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`s. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:
                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].

        Returns:
            results (list[Instances]): length `N` list of `Instances`s containing the
                detected instances. Returned during inference only; may be []
                during training.
            losses (dict[str: Tensor]): mapping from a named loss to a tensor
                storing the loss. Used during training only.
        """
        raise NotImplementedError()


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where the heads share the
    cropping and the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        assert len(self.in_features) == 1

        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / self.feature_strides[self.in_features[0]], )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.res5, out_channels = self._build_res5_block(cfg)
        output_layer = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg, out_channels, self.num_classes, self.cls_agnostic_bbox_reg
        )

    def _build_res5_block(self, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = make_stage(
            BottleneckBlock,
            3,
            first_stride=2,
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        return self.res5(x)

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images

        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        pred_class_logits, pred_proposal_deltas = self.box_predictor(
            feature_pooled
        )
        del feature_pooled

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )

        if self.training:
            del features
            losses = outputs.losses()
            return [], losses
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            return pred_instances, {}


@ROI_HEADS_REGISTRY.register()
class StandardROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(StandardROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg)

    def _init_box_head(self, cfg):
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels,
                height=pooler_resolution,
                width=pooler_resolution,
            ),
        )
        output_layer = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg,
            self.box_head.output_size,
            self.num_classes,
            self.cls_agnostic_bbox_reg,
        )

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        features_list = [features[f] for f in self.in_features]

        if self.training:
            losses = self._forward_box(features_list, proposals)
            return proposals, losses
        else:
            pred_instances = self._forward_box(features_list, proposals)
            return pred_instances, {}

    def _forward_box(self, features, proposals):
        """
        Forward logic of the box prediction branch.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = self.box_pooler(
            features, [x.proposal_boxes for x in proposals]
        )
        box_features = self.box_head(box_features)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(
            box_features
        )
        del box_features

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )
        if self.training:
            return outputs.losses()
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            return pred_instances


@ROI_HEADS_REGISTRY.register()
class StandardROIMultiHeads(StandardROIHeads):
    """
    Same as StandardROIHeads but allows for using multiple heads (e.g. different heads for base classes and novel
    classes)
    """
    def __init__(self, cfg, input_shape):
        super(StandardROIMultiHeads, self).__init__(cfg, input_shape)

    def _init_box_head(self, cfg):
        # fmt: off
        self.cpu_device = torch.device("cpu")
        self.device = torch.device(cfg.MODEL.DEVICE)
        pooler_resolution     = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales         = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio        = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type           = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        self.num_head_classes = cfg.MODEL.ROI_HEADS.MULTIHEAD_NUM_CLASSES  # classes per head
        self.num_heads        = cfg.MODEL.ROI_BOX_HEAD.NUM_HEADS
        # Dataset names because we need the appropriate metadata to obtain the correct class indices for each head!
        self.train_dataset_name = cfg.DATASETS.TRAIN[0]
        self.test_dataset_name = cfg.DATASETS.TEST[0]
        # fmt: on

        assert self.num_classes == sum(self.num_head_classes)
        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(  # TODO: probably force 'FastRCNNConvFCMultiHead'?
            cfg,
            ShapeSpec(
                channels=in_channels,
                height=pooler_resolution,
                width=pooler_resolution,
            ),
        )
        output_layer = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER

        self.box_predictors = []
        bbox_head_output_size = self.box_head.output_size
        if self.num_heads > 1:
            bbox_head_output_size //= self.num_heads
        for i in range(self.num_heads):
            box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
                    cfg,
                    bbox_head_output_size,
                    self.num_head_classes[i],
                    self.cls_agnostic_bbox_reg,
            )
            self.add_module("box_predictor{}".format(i+1), box_predictor)
            self.box_predictors.append(box_predictor)

    def _get_ind_mappings(self) -> List[Dict]:
        # Target indices range from 0 to 'cfg.MODEL.ROI_HEADS.NUM_CLASSES', but we here need, for each head i:
        #  a mapping from old index to range 0 to 'cfg.MODEL.ROI_HEADS.MULTIHEAD_NUM_CLASSES[i]'
        # Expected output: List(dict(int:int)), the list is expected to have one dict per head. Each dict is expected to
        #  map the large index of a class (from the single head) to the index used on this small head
        # Note: don't forget (for each head!) to map the background class (last index, not index 0!) to the last index
        # of this head's classes! (use self.num_head_classes[i] to access the amount of classes for head i)
        raise NotImplementedError

    def _forward_box(self, features, proposals):
        """
        Forward logic of the box prediction branch.
        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".
        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        # pooled features, result size is (e.g. [512, 256, 7, 7])
        # [MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
        # MODEL.FPN.OUT_CHANNELS?,
        # MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION,
        # MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION]
        box_features = self.box_pooler(
            features, [x.proposal_boxes for x in proposals]
        )
        # class-agnostic per-roi feature vectors, same size for each head
        # result is a list with '#heads' elements each of size
        #  [ROI_HEADS.BATCH_SIZE_PER_IMAGE * SOLVER.IMS_PER_BATCH, MODEL.ROI_BOX_HEAD.FC_DIM], e.g. [8192, 1024]
        box_features = self.box_head(box_features)
        assert len(box_features) == len(self.box_predictors) == self.num_heads, \
            "box_features output should match the amount of box predictors: {}, {}"\
            .format(len(box_features), len(self.box_predictors))

        # class-dependent logits and bbox deltas
        class_logits, proposal_deltas = [], []
        for i, box_predictor in enumerate(self.box_predictors):
            # pred_class_logits = [ROI_HEADS.BATCH_SIZE_PER_IMAGE * SOLVER.IMS_PER_BATCH, num_classes + 1]
            # pred_proposal_deltas =
            #  class-agnostic:  [ROI_HEADS.BATCH_SIZE_PER_IMAGE * SOLVER.IMS_PER_BATCH, 4]
            #  non cag:         [ROI_HEADS.BATCH_SIZE_PER_IMAGE * SOLVER.IMS_PER_BATCH, 4 x num_classes] Note: not num_classes + 1!
            pred_class_logits, pred_proposal_deltas = box_predictor(box_features[i])

            class_logits.append(pred_class_logits)
            proposal_deltas.append(pred_proposal_deltas)
        del box_features

        # Assumptions:
        # - 'box_features'-output from box_head is class-agnostic (ans same-sized for each head!), we can't do anything
        #    there!
        # - we use those features to obtain class-dependent activations (correct amount of target size is ensured by
        #    each 'predictor head'!
        # - for softmax calculation, we have to compare those activations against targets, which we obtain from
        #    the variable 'proposals', which contains objectness score (from RPN) and gt-class (and gt-boxes)
        # - those gt-data from the variable 'proposals' uses all available classes (and thus indices from
        #    0 to num_classes), we then need to transform those indices to appropriate indices for each head (and need
        #    to remember which number we mapped to which number at which head because each single head expects numbers
        #    or indices starting by 0, so our mapping destroys the unique numbers!
        # - we now have multiple possibilities what to do with our proposals: first of all, we decide to merge classes
        #    after softmax and to not merge the activations before the softmax. This would allow to skip the
        #    index-mapping but would also cause another problem: since each head produces background logits ans the
        #    final head, applying softmax on activations of all classes together just expects a single background class,
        #    so which of the background activations to choose and which to discard? This is a non-trivial problem and
        #    because of that, we choose to first apply softmax to each head and then merging the resulting class
        #    probabilities. We now assume wlog (without loss of generality) that we have batch-size 16 and using
        #    512 rois per batch yielding 8192 rois per batch (after roi pooling)
        #    - we could now take the Proposals and split them, depending on the target classes. In addition to this
        #      technique, we would probably want to use the background class activations for each head. If we think this
        #      idea a while further, we note that splitting of proposals into different heads does not make sense.
        #      We first note that each head i itself produces [8192, num_classes[i] + 1] classification logits because
        #      each head obtains 8192 rois as input (because the classification head splits after roi pooling, therefore
        #      each head encounters the same amount of input). For that matter, we either have to remove objects
        #      belonging to non-target classes at both sides, at feature side (class and box logits from the predictor)
        #      and at proposal-side (proposals from the RPN where GT-class is known), while keeping background class
        #      logits at EACH head.
        #    - another, and possibly more sophisticated, yet more simple, approach would be to use all proposals for
        #      each head with a little need in modification: at each head, change the target-class (gt-class) of the
        #      proposals for non-target classes of this head (not counting background class!) to 0. This means,
        #      non-target classes equal the background class.
        #      (Note: at Detectron2, the Background class is not the class with first index (0), but the class with
        #      last index (|num_classes|)!)
        #      Note: For training, we don't have to transform the indices back to the original indices because we're
        #      just interested in the loss which is automatically calculated correctly since the produced logits are
        #      yet in the correct shape and the adjusted class indices are automatically transferred into one-hot
        #      vectors for the classification loss (e.g. Cross Entropy). Therefore, we do not need back-transformation
        #      because we're done after calculating the loss.
        # - Inference: In contrast to training, we (of course) have not gt-annotations, therefore we cannot prepare or
        #   adjust the class of proposals. We don't even have to because we don't want to calculate losses. In contrast
        #   to the training however, we now need postprocessing of predicted classes after having calculated softmax
        #   probabilities because we need to know which class belongs to the highest probability for each proposal.
        #   In contrast to single-heads, we now have #heads predictions for each proposal because we input ALL
        #   proposals to each head. This could be problematic if we think of a case where for a single proposal one
        #   head produces a medium high confidence for an actual class (not background) and another head outputs high
        #   background confidence for that proposal (because it learned the target classes from different head as
        #   background class for itself). Probably this problem isn't an actual issue because the "Fast-RCNN"-Head
        #   won't output bbox predictions for Background class which would leave us with just a single valid prediction
        #   for that proposal (with medium confidence).

        # Proposals: contains 'SOLVER.IMS_PER_BATCH' elements of type detectron2.structures.Instances
        #   Access elements of list 'proposals' with indices.
        #   Access the elements of 'Instances' with '.get_fields()', or directly with '.tensor'
        #   Access the tensor (which 'Boxes' wraps) with boxes.tensor
        #     Note: Boxes supports __getitem__ using slices, indices, boolean arrays, etc.)
        #   Each Instance contains following fields of following sizes:
        #     'proposal_boxes': detectron2.structures.Boxes(tensor) of size [MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, 4]
        #     'objectness_logits': tensor of size [MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE]
        #     'gt_classes': tensor of size [MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE]
        #     'gt_boxes': detectron2.structures.Boxes(tensor) of size [MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, 4]
        # 'Proposals': (objectness logits + gt classes)
        # 'box_features': (pooled roi feature put forward though the net)

        # Algorithm:
        # (Synopsis: we use ALL proposals for each head and just need to set non-target classes to 0 (==Background))
        # 1. For Training, for each head i
        heads_proposals = []
        if self.training:
            all_inds_to_head_inds_list = self._get_ind_mappings()
            # 1.1 For each head i
            for i in range(len(self.box_predictors)):
                # 1.1.1 Take a copy of all Proposals, take the target categories of head i
                # list of #ROI_HEADS.BATCH_SIZE_PER_IMAGE Proposal-objects, each comprising
                # ROI_HEADS.BATCH_SIZE_PER_IMAGE proposals
                tmp_proposals = copy.deepcopy(proposals)
                all_inds_to_head_inds = all_inds_to_head_inds_list[i]
                all_bg_cls = self.num_classes
                head_bg_cls = self.num_head_classes[i]
                assert all_bg_cls in all_inds_to_head_inds and all_inds_to_head_inds[all_bg_cls] == head_bg_cls
                head_targets = list(all_inds_to_head_inds.keys())
                # Note: as of 'fast_rcnn'-doc, [0, num_cls) are foreground and |num_cls| is background!
                for instances in tmp_proposals:
                    gt_classes = instances.gt_classes  # ==instances.get_fields()['gt_classes']
                    # 1.1.2 Set the class of the j-th proposal to background class if its not a target class
                    # TODO: not sure about copying the tensor to host memory but torch currently does not support
                    #  the 'isin' function on its own...
                    bg_indices = np.isin(gt_classes.to(self.cpu_device), head_targets, invert=True).nonzero()
                    # using "all classes" background class, which is later transformed to appropriate background
                    #  class for this head
                    gt_classes[bg_indices] = all_bg_cls
                    # 1.1.3 If proposal j is a proposal for a target class, transform its class to range
                    # [0, num_classes[i]]
                    # Note: apply_ may only be used for cpu-tensors!, so we have move it to cpu temporarily
                    # TODO: 'apply_' might be slow since it's not easily parallelisable
                    gt_classes = gt_classes.to(self.cpu_device)  # move to CPU temporarily
                    gt_classes.apply_(lambda x: all_inds_to_head_inds[x])  # apply_ works inplace!
                    instances.gt_classes = gt_classes.to(self.device)  # move back to GPU and override object attribute
                heads_proposals.append(tmp_proposals)
        else:
            # 2.1 Pass all proposals to all heads
            for i in range(len(self.box_predictors)):
                heads_proposals.append(proposals)

        # Initialize 'FastRCNNOutputs'-object, nothing more!
        heads_outputs = []
        for i in range(len(self.box_predictors)):
            heads_outputs.append(
                FastRCNNOutputs(
                    self.box2box_transform,
                    class_logits[i],
                    proposal_deltas[i],
                    heads_proposals[i],
                    self.smooth_l1_beta,
                )
            )

        if self.training:
            # calculate losses e.g.
            #  'softmax cross entropy' on pred_class_logits ("loss_cls": self.softmax_cross_entropy_loss())
            #  'smooth L1 loss' on pred_proposal_deltas ("loss_box_reg": self.smooth_l1_loss())
            # Note: we don't need to transform any classes back to previous range because we're just interested in the
            #  loss. The gt-class (index in range of each head's target classes) will be automatically transformed to a
            #  one-hot vector which is sufficient to calculate the loss at each output neuron for each target class.
            #  We would just need to transform the categories back of we were interested in the name of each detection's
            #  class (as we are for inference).
            losses_dicts = {}
            for i, outputs in enumerate(heads_outputs):
                losses_dict = outputs.losses()
                for k, v in losses_dict.items():
                    losses_dicts[str(k) + "_" + str(i+1)] = v
                del losses_dict
            return losses_dicts
        else:
            pred_instances = []
            all_inds_to_head_inds_list = self._get_ind_mappings()
            for i, outputs in enumerate(heads_outputs):
                tmp_pred_instances, _ = outputs.inference(
                    self.test_score_thresh,
                    self.test_nms_thresh,
                    self.test_detections_per_img,  # TODO: problem in multi-head: detections_per_image_per_head?
                )
                # 2.2 After softmax, transform class of proposals back to range [0, all_classes]
                all_inds_to_head_inds = all_inds_to_head_inds_list[i]
                head_ind_to_ind = {v: k for k, v in all_inds_to_head_inds.items()}
                # 'tmp_pred_instances' is a list of 'Instances'-objects, one object for each image
                for instances in tmp_pred_instances:
                    # Note: at inference, this method is called once for each image, thus, |proposals| == 1
                    #  probably it is problematic to add one 'Instances'-object per head since the returned list has
                    #  twice the size as expected, probably, the remaining objects in the list are ignored!
                    # slow but ok for inference.
                    pred_classes = instances.pred_classes.to(self.cpu_device)  # move to cpu because of method 'apply_'
                    pred_classes.apply_(lambda x: head_ind_to_ind[x])  # element-wise inplace transformation
                    instances.pred_classes = pred_classes.to(self.device)  # move back to gpu and set object attribute
                pred_instances.append(tmp_pred_instances)
            # num images == len(proposals), where 'proposals' is the same in the list 'heads_proposals'
            # pred_instances = [num_heads, num_images], but we need [num images]
            #  [num_heads, num_images] -> [num_images, num_heads], then concatenate all 'Instances'-objects for a single
            #  image
            return [Instances.cat(list(x)) for x in zip(*pred_instances)]


@ROI_HEADS_REGISTRY.register()
class StandardROIDoubleHeads(StandardROIMultiHeads):
    """
    Same as StandardROIMultiHeads but using exactly two heads (for base classes and novel classes)
    """
    def __init__(self, cfg, input_shape):
        super(StandardROIDoubleHeads, self).__init__(cfg, input_shape)
        assert self.num_heads == 2, "To use Double-Head set num_heads to 2!"
        assert self.box_head.split_at_fc == 2, \
            "Current ckpt_surgery requires a fixed amount of fc layers as well as a firm split index of 2!"

    def _get_ind_mappings(self):
        dataset = self.train_dataset_name if self.training else self.test_dataset_name  # classes should normally be the same...
        metadata = MetadataCatalog.get(dataset)
        # For now, we use this kind of head solely for fine-tuning
        assert hasattr(metadata, 'novel_dataset_id_to_contiguous_id')
        assert hasattr(metadata, 'base_dataset_id_to_contiguous_id')
        all_id_to_inds = metadata.thing_dataset_id_to_contiguous_id
        base_id_to_inds = metadata.base_dataset_id_to_contiguous_id
        novel_id_to_inds = metadata.novel_dataset_id_to_contiguous_id
        all_inds_to_base_inds = {v: base_id_to_inds[k] for k, v in all_id_to_inds.items() if k in base_id_to_inds.keys()}
        all_inds_to_novel_inds = {v: novel_id_to_inds[k] for k, v in all_id_to_inds.items() if k in novel_id_to_inds.keys()}
        # For each head, add a mapping from old background class index to each head's background class index
        all_bg_ind = len(all_id_to_inds)
        base_bg_ind = len(base_id_to_inds)
        novel_bg_ind = len(novel_id_to_inds)
        assert all_bg_ind not in all_id_to_inds.values()
        assert base_bg_ind not in base_id_to_inds.values()
        assert novel_bg_ind not in novel_id_to_inds.values()
        all_inds_to_base_inds[all_bg_ind] = base_bg_ind
        all_inds_to_novel_inds[all_bg_ind] = novel_bg_ind
        return [all_inds_to_base_inds, all_inds_to_novel_inds]
