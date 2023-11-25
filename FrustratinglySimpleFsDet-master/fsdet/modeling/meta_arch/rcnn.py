import torch
from torch import nn

from fsdet.modeling.roi_heads import build_roi_heads

import logging
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n

# avoid conflicting with the existing GeneralizedRCNN module in Detectron2
from .build import META_ARCH_REGISTRY

__all__ = ["GeneralizedRCNN", "ProposalNetwork"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(
            cfg, self.backbone.output_shape()
        )
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = (
            torch.Tensor(cfg.MODEL.PIXEL_MEAN)
            .to(self.device)
            .view(num_channels, 1, 1)
        )
        pixel_std = (
            torch.Tensor(cfg.MODEL.PIXEL_STD)
            .to(self.device)
            .view(num_channels, 1, 1)
        )
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("froze backbone parameters")

        if cfg.MODEL.PROPOSAL_GENERATOR.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print("froze proposal generator parameters")

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            # keep this case for backwards-compatibility:
            # In old version, 'ROI_BOX_HEAD.FREEZE_*'-configs did not exist:
            #  base-training leaves all freeze configs at default values, which is no freezing at all
            #  fine-tuning always uses MODEL.ROI_HEADS.FREEZE_FEAT=True which leads to freezing of all
            #   roi box heads parameters
            # In new version, 'ROI_HEADS.FREEZE_FEAT' is never used:
            #  base-training also leaves freeze configs at default values which defaults to no freezing
            #  fine-tuning sets 'ROI_BOX_HEAD'-configs and leaves 'ROI_HEADS.FREEZE_FEAT' at False
            for p in self.roi_heads.box_head.parameters():
                p.requires_grad = False
            print("froze roi_box_head parameters")
        elif cfg.MODEL.ROI_HEADS.NAME == 'StandardROIDoubleHeads' and \
                cfg.MODEL.ROI_BOX_HEAD.NAME == 'FastRCNNConvFCMultiHead':
            # Custom freezing options for fine-tuning of a model with two heads (first head for base-classes and second
            #  head for novel classes), where the first head and its fc layers will be completely frozen (even
            #  classification and bbox regression!) and for the second head, the last fc-layer will remain unfrozen.
            #  This setting should allow for maintaining the base class performance while allowing the novel classes to
            #  be learned well.
            # freeze all 'conv', 'fc1' and 'fc2:1' parameters of the box head
            for k, v in self.roi_heads.box_head.named_modules():
                # We hard-code freezing of 'fc1' and 'fc2:1' because the class 'StandardROIDoubleHeads' ensures that
                #  we have exactly two heads and that we split the head always at index 2!
                if ('conv' in k) or ('fc1' in k) or ('fc2:1' in k):
                    for p in v.parameters():
                        p.requires_grad = False
                    print("Froze parameters of roi_box_head {} module".format(k))
            # additionally freeze the first predictor completely!
            for k, v in self.roi_heads.box_predictors[0].named_modules():
                # We explicitly name the modules for more informative messages
                if (k == 'cls_score') or (k == 'bbox_pred'):
                    for p in v.parameters():
                        p.requires_grad = False
                    print("Froze parameters of roi_box_predictor_1 {} module".format(k))
        else:
            # Freeze ROI BBOX Head Parameters
            name_to_module = {k: v for k, v in self.roi_heads.box_head.named_modules()}
            # could also use self.roi_heads.box_head.conv_norm_relus but we think of this solution as being more secure
            for conv_id in cfg.MODEL.ROI_BOX_HEAD.FREEZE_CONVS:
                assert 0 < conv_id <= len(cfg.MODEL.ROI_BOX_HEAD.FREEZE_CONVS)
                assert len(cfg.MODEL.ROI_BOX_HEAD.FREEZE_CONVS) <= cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
                conv_name = 'conv{}'.format(conv_id)
                assert conv_name in name_to_module
                for p in name_to_module[conv_name].parameters():
                    p.requires_grad = False
                print("froze roi_box_head {} parameters".format(conv_name))
            for fc_id in cfg.MODEL.ROI_BOX_HEAD.FREEZE_FCS:
                assert 0 < fc_id <= len(cfg.MODEL.ROI_BOX_HEAD.FREEZE_FCS)
                assert len(cfg.MODEL.ROI_BOX_HEAD.FREEZE_FCS) <= cfg.MODEL.ROI_BOX_HEAD.NUM_FC
                fc_name = 'fc{}'.format(fc_id)
                assert fc_name in name_to_module
                for p in name_to_module[fc_name].parameters():
                    p.requires_grad = False
                print("froze roi_box_head {} parameters".format(fc_name))

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs
            ]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN,
                "'targets' in the model inputs is now renamed to 'instances'!",
                n=10,
            )
            gt_instances = [
                x["targets"].to(self.device) for x in batched_inputs
            ]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [
                x["proposals"].to(self.device) for x in batched_inputs
            ]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(
            images, features, proposals, gt_instances
        )

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
        self, batched_inputs, detected_instances=None, do_postprocess=True
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [
                    x["proposals"].to(self.device) for x in batched_inputs
                ]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [
                x.to(self.device) for x in detected_instances
            ]
            results = self.roi_heads.forward_with_given_boxes(
                features, detected_instances
            )

        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(
            images, self.backbone.size_divisibility
        )
        return images


@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(
            cfg, self.backbone.output_shape()
        )

        pixel_mean = (
            torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(-1, 1, 1)
        )
        pixel_std = (
            torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(-1, 1, 1)
        )
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]: Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(
            images, self.backbone.size_divisibility
        )
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs
            ]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN,
                "'targets' in the model inputs is now renamed to 'instances'!",
                n=10,
            )
            gt_instances = [
                x["targets"].to(self.device) for x in batched_inputs
            ]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances
        )
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results
