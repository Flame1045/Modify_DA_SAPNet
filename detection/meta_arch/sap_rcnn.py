import torch.nn as nn
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from detectron2.config import configurable
from detectron2.structures import Instances
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling import Backbone, build_backbone, build_proposal_generator, build_roi_heads, META_ARCH_REGISTRY
from detectron2.utils.events import get_event_storage
from ..da_heads import build_DAHead
from ..da_heads import Masking
from ..da_heads import EMATeacher


@META_ARCH_REGISTRY.register()
class SAPRCNN(GeneralizedRCNN):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        da_heads: Union[nn.Module, None],
        in_feature_da_heads: str = 'p6',
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        super().__init__(backbone=backbone, proposal_generator=proposal_generator, roi_heads=roi_heads, \
            pixel_mean=pixel_mean, pixel_std=pixel_std, input_format=input_format, vis_period=vis_period, \
        )
        self.da_heads = da_heads
        self.in_feature_da_heads = in_feature_da_heads

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        if cfg.MODEL.DOMAIN_ADAPTATION_ON:
            da_haeds = build_DAHead(cfg)
        else:
            da_haeds = None
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "in_feature_da_heads": cfg.MODEL.DA_HEAD.IN_FEATURE,
            "da_heads": da_haeds,
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    def forward(self, source_batched_inputs: List[Dict[str, torch.Tensor]]=None, target_batched_inputs:List[Dict[str, torch.Tensor]]=None,
                mt_batched_inputs: List[Dict[str, torch.Tensor]]=None, cfg=None, masking=None, pseudo_gt=None, pseduo_flag=False, losses=None):
        """
        training flow
        Args:
            source_batched_inputs, target_batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
            input_domain: str, source or target domain input 
        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        # masking image 
        # if masking is not None:
        #     return self.preprocess_image(target_batched_inputs).tensor                  
        if not self.training and pseduo_flag is False:
            return self.inference(source_batched_inputs)
        # source domain input
        if source_batched_inputs is not None:
            s_images = self.preprocess_image(source_batched_inputs)
            if "instances" in source_batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in source_batched_inputs]
            else:
                gt_instances = None 
            s_features = self.backbone(s_images.tensor)
            # s_features, s_images, gt_instances

        # target domain input, proposal_generator == RPN
        if self.da_heads:
            if target_batched_inputs is not None:
                t_images = self.preprocess_image(target_batched_inputs)
                t_features = self.backbone(t_images.tensor)
            if cfg.MODEL.DA_HEAD.MIC_ON:
                if pseudo_gt is not None:
                    mt_images = self.preprocess_image(mt_batched_inputs)
                    masked_images = masking(mt_images.tensor)
                    masked_features = self.backbone(masked_images) 
                    mt_proposals, mt_proposal_losses, mt_rpn_logits = self.proposal_generator(mt_images, masked_features, pseudo_gt, mask_flag=True)
                    mt_proposal_losses['loss_mic_rpn_cls'] = mt_proposal_losses['loss_rpn_cls'].clone()
                    mt_proposal_losses['loss_mic_rpn_loc'] = mt_proposal_losses['loss_rpn_loc'].clone() 
                    del mt_proposal_losses['loss_rpn_cls'], mt_proposal_losses['loss_rpn_loc']
                    _, mt_detector_losses = self.roi_heads(mt_images, masked_features, mt_proposals, pseudo_gt)
                    mt_detector_losses['loss_mic_cls'] = mt_detector_losses['loss_cls'].clone()
                    mt_detector_losses['loss_mic_box_reg'] = mt_detector_losses['loss_box_reg'].clone()
                    del mt_detector_losses['loss_cls'], mt_detector_losses['loss_box_reg']
                    # mic_losses = {'mic_Loss': mt_proposal_losses['loss_mic_rpn_cls'] + \
                    #                 mt_proposal_losses['loss_mic_rpn_loc'] + \
                    #                 mt_detector_losses['loss_mic_cls'] + \
                    #                 mt_detector_losses['loss_mic_box_reg']
                    #              }

                    # losses.update(mt_proposal_losses)
                    # return losses
                if pseduo_flag:
                    tmp = self.inference(target_batched_inputs)
                    # t_images_output, _, _ = self.proposal_generator(t_images, t_features, None)
                    t_images_output = []
                    for i in range(len(tmp)):
                        t_images_output.append(tmp[i]['instances'])
                    return t_images_output
            _, medm_loss, t_rpn_logits = self.proposal_generator(t_images, t_features, None)
            s_proposals, proposal_losses, s_rpn_logits = self.proposal_generator(s_images, s_features, gt_instances)
            da_source_loss = self.da_heads(s_features[self.in_feature_da_heads], s_rpn_logits, 'source')
            da_target_loss = self.da_heads(t_features[self.in_feature_da_heads], t_rpn_logits, 'target')
        else:
            if self.proposal_generator is not None:
                s_proposals, proposal_losses, s_rpn_logits = self.proposal_generator(s_images, s_features, gt_instances)
            else:
                assert "proposals" in source_batched_inputs[0]
                s_proposals = [x["proposals"].to(self.device) for x in source_batched_inputs]
                proposal_losses = {}

        if source_batched_inputs is not None:
            _, detector_losses = self.roi_heads(s_images, s_features, s_proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(source_batched_inputs, s_proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        if self.da_heads:
            losses.update(da_source_loss)
            losses.update(da_target_loss)
            if medm_loss:
                losses.update(medm_loss)
        if pseudo_gt is not None:
            losses.update(mt_proposal_losses)
            losses.update(mt_detector_losses)
            # losses.update(mic_losses)
        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
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
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)


        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def visualize_spatial_attention_mask(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        '''
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
            output: dict[str, torch.Tensor], file name + index of spatial mask, and corresponding tensor 
        '''
        assert not self.training
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        anchor_logits = self.proposal_generator.get_anchor_classfication_logit(features)
        attention_masks = self.da_heads.spatial_attention_mask(features[self.in_feature_da_heads], anchor_logits)
        output = {}
        for idx, d in enumerate(batched_inputs):
            name = Path(d['file_name']).stem
            for j, sp in enumerate(attention_masks):
                output[name + f'-{j}'] = sp[idx]
        return output

    def get_domain_vector(self, batched_inputs: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        '''
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
            torch.Tensor, semantic vector
        '''
        assert not self.training
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        anchor_logits = self.proposal_generator.get_anchor_classfication_logit(features)
        return self.da_heads.logit_vector(features[self.in_feature_da_heads], anchor_logits)