import torch.nn as nn
import torch
import cv2
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
import random
import numpy
import torch.nn.functional as F
from torch import nn
from detectron2.structures.boxes import BoxMode
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.structures import ImageList, Instances, Boxes
from ..modeling.bounding_box import BoxList


def setup_seed(seed):
    random.seed(seed)                          
    numpy.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

class MaxSquareloss(nn.Module):
    def __init__(self, ignore_index= -1, num_class=1):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_class = num_class
    
    def forward(self, prob):
        """
        :param pred: predictions (N, C, H, W)
        :param prob: probability of pred (N, C, H, W)
        :return: maximum squares loss
        """
        # prob -= 0.5
        mask = (prob != self.ignore_index)    
        loss = -torch.mean(torch.pow(prob, 2)[mask]) / 2
        return loss

class ResidualConnet(nn.Module):
    def __init__(self):
        super(ResidualConnet, self).__init__()
        self.shortcut = nn.Identity()
    
    def forward(self, x1, x2):
        residual = x1  # Store the input for the shortcut connection
        # Add the shortcut connection
        out = self.shortcut(residual) + x2
        out = torch.relu(out)
        
        return out

@META_ARCH_REGISTRY.register()
class SAPRCNN_ORALCLE(GeneralizedRCNN):
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
        self.MSLoss = MaxSquareloss(num_class=1)
        self.resconnet = ResidualConnet()

    @classmethod
    def from_config(cls, cfg):
        setup_seed(cfg.SEED) 
        print("sap_rcnn SAPRCNN from_config seeding")  
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
                mt_batched_inputs: List[Dict[str, torch.Tensor]]=None, cfg=None, masking=None, pseudo_gt=None, pseudo_gt2=None,
                pseduo_flag=False, losses=None):
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
        # setup_seed(cfg.SEED) 
        # print("sap_rcnn SAPRCNN forward seeding")          
        if not self.training and pseduo_flag is False:
            return self.inference(source_batched_inputs, cfg = None)
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
                if "instances" in target_batched_inputs[0]:
                    target_gt_instances = [x["instances"].to(self.device) for x in target_batched_inputs]
                t_images = self.preprocess_image(target_batched_inputs)
                t_features = self.backbone(t_images.tensor)
            if cfg.MODEL.DA_HEAD.MIC_ON:
                if pseudo_gt is not None:
                    mt_images = self.preprocess_image(mt_batched_inputs)
                    masked_images = masking(mt_images.tensor)
                    masked_features = self.backbone(masked_images) 
                    mt_proposals, mt_proposal_losses, mt_rpn_logits = self.proposal_generator(mt_images, masked_features, pseudo_gt, mask_flag=True, cfg=cfg)
                    mt_proposal_losses['loss_mic_rpn_cls'] = mt_proposal_losses['loss_rpn_cls'].clone()
                    mt_proposal_losses['loss_mic_rpn_loc'] = mt_proposal_losses['loss_rpn_loc'].clone() 
                    del mt_proposal_losses['loss_rpn_cls'], mt_proposal_losses['loss_rpn_loc']
                    _, mt_detector_losses = self.roi_heads(mt_images, masked_features, mt_proposals, pseudo_gt)
                    mt_detector_losses['loss_mic_cls'] = mt_detector_losses['loss_cls'].clone()
                    mt_detector_losses['loss_mic_box_reg'] = mt_detector_losses['loss_box_reg'].clone()
                    del mt_detector_losses['loss_cls'], mt_detector_losses['loss_box_reg']
                if pseduo_flag:
                    t_images_output = self.inference(target_batched_inputs, do_postprocess = False, is_pseudo=True, cfg = None)
                    return t_images_output 
                                      
            if cfg.MODEL.DA_HEAD.ILLUME:
                _, medm_loss, t_rpn_logits = self.proposal_generator(t_images, t_features, cfg=cfg)
                s_proposals, proposal_losses, s_rpn_logits = self.proposal_generator(s_images, s_features, gt_instances, cfg=cfg)

                da_source_loss, ILLUME_source_features = self.da_heads(s_features[self.in_feature_da_heads], s_rpn_logits, 'source')
                s_features[self.in_feature_da_heads] = self.resconnet(ILLUME_source_features, s_features[self.in_feature_da_heads])

                da_target_loss, ILLUME_target_features = self.da_heads(t_features[self.in_feature_da_heads], t_rpn_logits, 'target') 
                t_features[self.in_feature_da_heads] = self.resconnet(ILLUME_target_features, t_features[self.in_feature_da_heads])

            else:
                t_proposals, t_proposal_losses, t_rpn_logits = self.proposal_generator(t_images, t_features, target_gt_instances, cfg=cfg)  ###
                s_proposals, proposal_losses, s_rpn_logits = self.proposal_generator(s_images, s_features, gt_instances, cfg=cfg)
                da_source_loss = self.da_heads(s_features[self.in_feature_da_heads], s_rpn_logits, 'source')
                da_target_loss = self.da_heads(t_features[self.in_feature_da_heads], t_rpn_logits, 'target')
                t_MSLoss, s_MSLoss = 0.0, 0.0

            if cfg.MODEL.FINETUNE_PSEUDO_FINETUNE_PSEUDO_ON:
                for t_rpn_logit in t_rpn_logits:
                    t_rpn_logit =  torch.sigmoid(t_rpn_logit)
                    t_MSLoss = t_MSLoss + self.MSLoss(t_rpn_logit)
                for s_rpn_logit in s_rpn_logits:
                    s_rpn_logit = torch.sigmoid(s_rpn_logit)
                    s_MSLoss = s_MSLoss + self.MSLoss(s_rpn_logit)
        else:
            if self.proposal_generator is not None:
                s_proposals, proposal_losses, s_rpn_logits = self.proposal_generator(s_images, s_features, gt_instances, cfg=cfg)
            else:
                assert "proposals" in source_batched_inputs[0]
                s_proposals = [x["proposals"].to(self.device) for x in source_batched_inputs]
                proposal_losses = {}

        if source_batched_inputs is not None:
            _, detector_losses = self.roi_heads(s_images, s_features, s_proposals, gt_instances)
        if target_batched_inputs is not None:
            _, target_detector_losses = self.roi_heads(t_images, t_features, t_proposals, target_gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(source_batched_inputs, s_proposals)

        # print('target_detector_losses:',target_detector_losses)
        # print('t_proposal_losses:',t_proposal_losses)

        t_proposal_losses['ORALCLE_loss_rpn_cls'] = t_proposal_losses['loss_rpn_cls'].clone()
        t_proposal_losses['ORALCLE_loss_rpn_loc'] = t_proposal_losses['loss_rpn_loc'].clone() 
        target_detector_losses['ORALCLE_loss_cls'] = target_detector_losses['loss_cls'].clone()
        target_detector_losses['ORALCLE_loss_box_reg'] = target_detector_losses['loss_box_reg'].clone()
        del target_detector_losses['loss_cls'], target_detector_losses['loss_box_reg']
        del t_proposal_losses['loss_rpn_cls'], t_proposal_losses['loss_rpn_loc']

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        if self.da_heads:
            losses.update(da_source_loss)
            losses.update(da_target_loss)
            # if medm_loss:
            #     losses.update(medm_loss)
        if pseudo_gt is not None:
            losses.update(mt_proposal_losses)
            losses.update(mt_detector_losses)
            # losses.update(mic_losses)
        if cfg.MODEL.FINETUNE_PSEUDO_FINETUNE_PSEUDO_ON:
            losses.update({'loss_target_MS': t_MSLoss,
                           'loss_source_MS': s_MSLoss})
        if cfg.MODEL.ORALCLE:
            losses.update(target_detector_losses)
            losses.update(t_proposal_losses)
        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
        is_pseudo = False,
        cfg = None
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
        # setup_seed(cfg.SEED)
        # print("sap_rcnn SAPRCNN inference seeding!")
        # print("META ARCH")
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        # images.tensor().cpu()
        # features.tensor().cpu()

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _1, _2 = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            if is_pseudo:
                results, _3 = self.roi_heads(images, features, proposals, None, is_pseudo=True) ###MIC
            else:
                results, _3 = self.roi_heads(images, features, proposals, None)
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

def Adv_GRL(loss_iter, input_features, list_option=True, cfg=None):

        bce = F.binary_cross_entropy_with_logits(torch.FloatTensor([[0.7,0.3]]), torch.FloatTensor([[1,0]]))
        GradientScalarLayer = GradientScalarLayer_()

        if loss_iter <=  bce:
            adv_threshold = min(cfg.advGRL_threshold, 1/loss_iter)
            # self.advGRL_optimized = GradientScalarLayer(-1.0*self.cfg.MODEL.DA_HEADS.DA_IMG_advGRL_WEIGHT*adv_threshold)

            if list_option:# for img_features (list[Tensor])
                advGRL_optimized = GradientScalarLayer(-1.0*cfg.MODEL.DA_HEADS.DA_IMG_advGRL_WEIGHT*adv_threshold.numpy())
                advgrl_fea = [ advGRL_optimized(fea) for fea in input_features]
            else: # for da_ins_feature (Tensor)
                advGRL_optimized = GradientScalarLayer(-1.0*cfg.MODEL.DA_HEADS.DA_INS_advGRL_WEIGHT*adv_threshold.numpy())
                advgrl_fea = advGRL_optimized(input_features)
        else:
            if list_option:# for img_features (list[Tensor])
                grl_img = GradientScalarLayer(-1.0*cfg.MODEL.DA_HEADS.DA_IMG_GRL_WEIGHT)
                advgrl_fea = [grl_img(fea) for fea in input_features]###the original component
            else: # for da_ins_feature (Tensor)
                grl_ins = GradientScalarLayer(-1.0*cfg.MODEL.DA_HEADS.DA_INS_GRL_WEIGHT)
                advgrl_fea = grl_ins(input_features)

        # print("Adv_GRL is used")
        
        return advgrl_fea

class GradientScalarLayer_(torch.nn.Module):
    def __init__(self, weight):
        super(GradientScalarLayer_, self).__init__()
        self.weight = weight

    def forward(self, input):
        return gradient_scalar(input, self.weight)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "weight=" + str(self.weight)
        tmpstr += ")"
        return tmpstr

class _GradientScalarLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.weight = weight
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return ctx.weight*grad_input, None

gradient_scalar = _GradientScalarLayer.apply