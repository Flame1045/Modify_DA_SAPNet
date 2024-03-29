# All the trainer support mixed precision training
import logging
import weakref
import time
import torch
import torch.nn.functional as F
from functools import partial
from detectron2.utils import comm
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer, create_ddp_model, SimpleTrainer, hooks
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.data import MetadataCatalog
import detectron2.data.transforms as T
from typing import Dict, List, Optional, Tuple
from torch.nn.parallel import DataParallel, DistributedDataParallel
from fvcore.nn.precise_bn import get_bn_modules
from torchvision.transforms.functional import center_crop as centcp
from .evaluation.pascal_voc import PascalVOCDetectionEvaluator_
from .data.build import build_DA_detection_train_loader
from .da_heads.masking import Masking
from .da_heads.teacher import EMATeacher
import numpy as np
import random
import cv2
from .modeling.bounding_box import BoxList
import sys
import os
import shutil


def setup_seed(seed):
    random.seed(seed)                          
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

class _DATrainer(SimpleTrainer):
    # one2one domain adpatation trainer
    def __init__(self, model, source_domain_data_loader, target_domain_data_loader, loss_weight, optimizer, cfg):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of losses.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
        """
        super(SimpleTrainer).__init__()

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        setup_seed(cfg.SEED)
        print("_DATrainer __init__ seeding")
        model.train()
        self.model = model
        self.source_domain_data_loader = source_domain_data_loader
        self.target_domain_data_loader = target_domain_data_loader
        self._source_domain_data_loader_iter = iter(source_domain_data_loader)
        self._target_domain_data_loader_iter = iter(target_domain_data_loader)
        self.loss_weight = loss_weight
        self.optimizer = optimizer
        self.cfg = cfg

    def run_step(self):
        assert self.model.training, "[_DATrainer] model was changed to eval mode!"

        start = time.perf_counter()
        s_data = next(self._source_domain_data_loader_iter)
        data_time = time.perf_counter() - start

        start = time.perf_counter()
        t_data = next(self._target_domain_data_loader_iter)
        data_time = time.perf_counter() - start + data_time
        # test = my_preprocess_image(self=self.model, batched_inputs=t_data)
        loss_dict = self.model(s_data, t_data, cfg=self.cfg)
        loss_dict = {l: self.loss_weight[l] * loss_dict[l] for l in self.loss_weight}
        losses = sum(loss_dict.values())
        self.optimizer.zero_grad()
        losses.backward()
        self._write_metrics(loss_dict, data_time)
        self.optimizer.step()

class _DATrainer_MIC(SimpleTrainer):
    # one2one domain adpatation trainer
    def __init__(self, model, teacher_model, masking, source_domain_data_loader, target_domain_data_loader, loss_weight, optimizer, cfg):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of losses.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
        """
        super(SimpleTrainer).__init__()

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        setup_seed(cfg.SEED)
        print("trainer _DATrainer_MIC __init__ seeding")
        model.train()
        self.model = model
        self.source_domain_data_loader = source_domain_data_loader
        self.target_domain_data_loader = target_domain_data_loader
        self._source_domain_data_loader_iter = iter(source_domain_data_loader)
        self._target_domain_data_loader_iter = iter(target_domain_data_loader)
        self.loss_weight = loss_weight
        self.optimizer = optimizer
        self.cfg = cfg
        self.model_teacher = teacher_model
        self.masking = masking
        self.iterations = 0
        

    def run_step(self):
        # setup_seed(self.cfg.SEED)
        # print("trainer _DATrainer_MIC run_step seeding")
        assert self.model.training, "[_DATrainer] model was changed to eval mode!"

        start = time.perf_counter()
        s_data = next(self._source_domain_data_loader_iter)
        data_time = time.perf_counter() - start

        start = time.perf_counter()
        t_data = next(self._target_domain_data_loader_iter)
        data_time = time.perf_counter() - start + data_time

        if self.cfg.MODEL.DA_HEAD.MIC_ON == True:
            self.model_teacher.update_weights(self.model, self.iterations)
            self.iterations += 1
            target_output = self.model_teacher(target_img=t_data, cfg=self.cfg)
            target_pseudo_labels, _ = process_pred2label(target_output, threshold=self.cfg.MODEL.DA_HEAD.PSEUDO_LABEL_THRESHOLD)
            self.model.train()
            loss_dict = self.model(source_batched_inputs=s_data, target_batched_inputs=t_data,
                        mt_batched_inputs=t_data, masking=self.masking, pseudo_gt=target_pseudo_labels, cfg=self.cfg)                     
        else:
            loss_dict = self.model(s_data, t_data, self.cfg)

        loss_dict = {l: self.loss_weight[l] * loss_dict[l] for l in self.loss_weight}
        losses = sum(loss_dict.values())
        self.optimizer.zero_grad()
        losses.backward()
        self._write_metrics(loss_dict, data_time)
        self.optimizer.step()

class DefaultPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.

    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.

    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more complicated, please refer to its source code as
    examples to build and use the model manually.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.

    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TRAIN):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.PSEUDO_WEIGHTS)
        print("Load PSEUDO WEIGHTS: ",  cfg.MODEL.PSEUDO_WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            final_predictions = []
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]

            # get predict boxes
            pred_bboxes = predictions['instances']._fields['pred_boxes'].tensor.detach()
            # get predict logits(which can be negative and need to be converted to probability))
            scores = predictions['instances']._fields['scores'].detach()
            # filter out low probability boxes, and its corresponding labels
            filtered_idx = scores>=0.85
            filtered_bboxes = pred_bboxes[filtered_idx]
            # create new BoxList, which is used to store filtered_bboxes(Tensor)
            new_bbox_list = BoxList(filtered_bboxes, predictions['instances']._image_size, mode="xyxy")
            # convert to gt_instances format(Instances)
            final_predictions = Instances(new_bbox_list.size)
            final_predictions.pred_boxes = Boxes(new_bbox_list.bbox)
            final_predictions.scores = predictions['instances']._fields['scores'][filtered_idx]
            final_predictions.pred_classes = predictions['instances']._fields['pred_classes'][filtered_idx]

            return final_predictions


class _DATrainer_Pseudo_gen(SimpleTrainer):
    # one2one domain adpatation trainer
    def __init__(self, model, source_domain_data_loader, target_domain_data_loader, loss_weight, optimizer, cfg):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of losses.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
        """
        super(SimpleTrainer).__init__()

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        setup_seed(cfg.SEED)
        print("trainer _DATrainer_MIC __init__ seeding")
        model.train()
        self.model = model
        self.source_domain_data_loader = source_domain_data_loader
        self.target_domain_data_loader = target_domain_data_loader
        self._source_domain_data_loader_iter = iter(source_domain_data_loader)
        self._target_domain_data_loader_iter = iter(target_domain_data_loader)
        self.loss_weight = loss_weight
        self.optimizer = optimizer
        self.cfg = cfg
        self.predictor = DefaultPredictor(cfg)
        

    def run_step(self):
        # setup_seed(self.cfg.SEED)
        # print("trainer _DATrainer_MIC run_step seeding")
        assert self.model.training, "[_DATrainer] model was changed to eval mode!"

        if self.cfg.MODEL.DA_HEAD.Pseudo_gen == True:
            self.model.eval()

            try:
                print('Annotations copying')
                shutil.copytree('datasets/Cityscapes-coco/VOC2007-car-train/Annotations', 
                            'datasets/Cityscapes-coco/pseudo_img_v2/Annotations',  dirs_exist_ok=True)
                print('Annotations copy finish!')
            except Exception as e:
                print('Directory not copied.')
                print(e)
                sys.exit(1) # Failed

            try:
                print('ImageSets txt copying')
                shutil.copytree('datasets/Cityscapes-coco/VOC2007-car-train/ImageSets', 
                                'datasets/Cityscapes-coco/pseudo_img_v2/ImageSets',  dirs_exist_ok=True)
                print('ImageSets txt copy finish!')
            except Exception as e:
                print('Directory not copied.')
                print(e)
                sys.exit(1) # Failed

            try:        
                print('JPEGImages copying (about 3 minutes)')
                # shutil.copytree('datasets/Cityscapes-coco/VOC2007-car-train/JPEGImages.zip', 
                #                 'datasets/Cityscapes-coco/pseudo_img_v2',  dirs_exist_ok=True)
                from tqdm import tqdm
                import zipfile
                with zipfile.ZipFile('datasets/Cityscapes-coco/VOC2007-car-train/JPEGImages.zip') as zf:
                    for member in tqdm(zf.infolist(), desc='Extracting '):
                        try:
                            zf.extract(member, 'datasets/Cityscapes-coco/pseudo_img_v2')
                        except zipfile.error as e:
                            pass
                print('JPEGImages copy finish!')
            except Exception as e:
                print('Directory not copied.')
                print(e)
                sys.exit(1) # Failed

            for i in range(0, 1000):
                print(i+1,"/1000")
                t_pseudo = []
                t_data = next(self._target_domain_data_loader_iter)
                for j in range(0,2):
                    im = cv2.imread(t_data[j].get('file_name'), cv2.IMREAD_COLOR)
                    t_pseudo.append(self.predictor(im))
                pseudo_dataset_gen(self.cfg, t_data, t_pseudo)

            sys.exit(0) # Success        
        else:
            start = time.perf_counter()
            s_data = next(self._source_domain_data_loader_iter)
            data_time = time.perf_counter() - start

            start = time.perf_counter()
            t_data = next(self._target_domain_data_loader_iter)
            data_time = time.perf_counter() - start + data_time
            loss_dict = self.model(source_batched_inputs=s_data, target_batched_inputs=t_data, cfg=self.cfg)

        loss_dict = {l: self.loss_weight[l] * loss_dict[l] for l in self.loss_weight}
        losses = sum(loss_dict.values())
        self.optimizer.zero_grad()
        losses.backward()
        self._write_metrics(loss_dict, data_time)
        self.optimizer.step()

class _DAAMPTrainer(_DATrainer):
    def __init__(self, model, source_domain_data_loader, target_domain_data_loader, loss_weight, optimizer, grad_scaler=None):
        
        unsupported = "_DAAMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported
        super().__init__(model, source_domain_data_loader, target_domain_data_loader, loss_weight, optimizer)
        if grad_scaler is None:
            from torch.cuda.amp import GradScaler
            grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler

    def run_step(self):
        """
        Implement the AMP training logic.
        """
        assert self.model.training, "[_DAAMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[_DAAMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        s_data = next(self._source_domain_data_loader_iter)
        data_time = time.perf_counter() - start

        start = time.perf_counter()
        t_data = next(self._target_domain_data_loader_iter)
        data_time = time.perf_counter() - start + data_time

        with autocast():
            loss_dict = self.model(s_data, t_data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                loss_dict = {l: self.loss_weight[l] * loss_dict[l] for l in self.loss_weight}
                losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        self.grad_scaler.scale(losses).backward()

        self._write_metrics(loss_dict, data_time)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

    def state_dict(self):
        ret = super().state_dict()
        ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.grad_scaler.load_state_dict(state_dict["grad_scaler"])

class DATrainer(DefaultTrainer):
    # one2one domain adpatation trainer
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """       

        setup_seed(cfg.SEED)
        print("trainer DATrainer __init__ seeding")  

        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        source_domain_data_loader = self.build_train_loader(cfg, 'source')
        target_domain_data_loader = self.build_train_loader(cfg, 'target')

        model = create_ddp_model(model, broadcast_buffers=False)
        loss_weight = {'loss_cls': 1, 'loss_box_reg': 1, 'loss_rpn_cls': 1, 'loss_rpn_loc': 1,\
        'loss_sap_source_domain': cfg.MODEL.DA_HEAD.LOSS_WEIGHT, 'loss_sap_target_domain': cfg.MODEL.DA_HEAD.LOSS_WEIGHT}

        if cfg.MODEL.DA_HEAD.RPN_MEDM_ON:
            loss_weight.update({'loss_target_entropy': cfg.MODEL.DA_HEAD.TARGET_ENT_LOSS_WEIGHT, 'loss_target_diversity': cfg.MODEL.DA_HEAD.TARGET_DIV_LOSS_WEIGHT})
        if cfg.MODEL.DA_HEAD.MIC_ON:
            # EMA model teacher
            model_t = self.build_model(cfg)
            model_t = create_ddp_model(model_t, broadcast_buffers=False)

            # loss_weight.update({'loss_mt_rpn_cls': 1, 'loss_mt_rpn_loc': 1, 'loss_mt_cls': 1, 'loss_mt_box_reg': 1})
            loss_weight.update({'loss_mic_rpn_cls': cfg.MODEL.DA_HEAD.MIC_RPN_CLS_WEIGHT, 
                                'loss_mic_rpn_loc': cfg.MODEL.DA_HEAD.MIC_RPN_LOC_WEIGHT, 
                                'loss_mic_cls': cfg.MODEL.DA_HEAD.MIC_CLS_WEIGHT,
                                'loss_mic_box_reg': cfg.MODEL.DA_HEAD.MIC_BOX_REG_WEIGHT,
                                })
            # loss_weight.update({'mic_Loss': 1})
            masking = Masking(
                block_size=cfg.MODEL.DA_HEAD.MASKING_BLOCK_SIZE,
                ratio=cfg.MODEL.DA_HEAD.MASKING_RATIO,
                color_jitter_s=cfg.MODEL.DA_HEAD.MASK_COLOR_JITTER_S, 
                color_jitter_p=cfg.MODEL.DA_HEAD.MASK_COLOR_JITTER_P, 
                blur=cfg.MODEL.DA_HEAD.MASK_BLUR,                      
                mean=cfg.MODEL.DA_HEAD.PIXEL_MEAN,                      
                std=cfg.MODEL.DA_HEAD.PIXEL_STD)
            teacher_model = EMATeacher(model_t, alpha=cfg.MODEL.DA_HEAD.TEACHER_ALPHA).to(model_t.device)
            teacher_model.eval()
            self._trainer = (_DATrainer_MIC)(
                model, teacher_model, masking, source_domain_data_loader, target_domain_data_loader, loss_weight, optimizer, cfg
            ) 
        if cfg.MODEL.FINETUNE_PSEUDO_FINETUNE_PSEUDO_ON:
            loss_weight.update({'loss_target_MS': cfg.MODEL.FINETUNE_PSEUDO_TARGET_WEIGHTS, 
                                'loss_source_MS': cfg.MODEL.FINETUNE_PSEUDO_SOURCE_WEIGHTS})
            
        if cfg.MODEL.ORALCLE:
            loss_weight.update({'ORALCLE_loss_cls': 1, 'ORALCLE_loss_box_reg': 1, 
                                'ORALCLE_loss_rpn_cls': 1, 'ORALCLE_loss_rpn_loc': 1,})

        if cfg.MODEL.DA_HEAD.Pseudo_gen:
           self._trainer = (_DATrainer_Pseudo_gen)(
                model, source_domain_data_loader, target_domain_data_loader, loss_weight, optimizer, cfg
           )
        else:
            self._trainer = (_DAAMPTrainer if cfg.SOLVER.AMP.ENABLED else _DATrainer)(
                model, source_domain_data_loader, target_domain_data_loader, loss_weight, optimizer, cfg
            )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())
    
    @classmethod
    def build_train_loader(cls, cfg, dataset_domain):
        return build_DA_detection_train_loader(cfg, dataset_domain=dataset_domain)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return PascalVOCDetectionEvaluator_(dataset_name)

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.LRScheduler(),
            hooks.IterationTimer(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]
        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    @classmethod
    def build_optimizer(cls, cfg, model):
        return DefaultTrainer_.build_optimizer(cfg, model)

class DefaultTrainer_(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return PascalVOCDetectionEvaluator_(dataset_name)

    @classmethod
    def build_optimizer(cls, cfg, model):
        if cfg.SOLVER.NAME == 'default':
            return super().build_optimizer(cfg, model)

        elif cfg.SOLVER.NAME == 'adam':
            return torch.optim.Adam(
                [p for name, p in model.named_parameters() if p.requires_grad], 
                lr=cfg.SOLVER.BASE_LR,
                betas=(0.9, 0.999), weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        elif cfg.SOLVER.NAME == 'adamw':
            return torch.optim.AdamW(
                [p for name, p in model.named_parameters() if p.requires_grad], 
                lr=cfg.SOLVER.BASE_LR,
                betas=(0.9, 0.999), weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError('Not support {}'.format(cfg.SOLVER.NAME))

class SpatialAttentionVisualHelper:
    def __init__(self, cfg, test_set):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        data_loader = DATrainer.build_test_loader(cfg, test_set)
        self.data_size = len(data_loader) # for `visualize_attention_mask()` in train_net.py to track dataloader's size
        self.dataloader = iter(data_loader)

    def __call__(self):
        """
        Returns:
            predictions (dict):
                the output of the model
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():
            data = next(self.dataloader)
            predictions = self.model.visualize_spatial_attention_mask(data)
            return predictions


class GramCamForDomainClassfier:
    '''
    Both domain run grad cam seperately, target layer is spatial attention mask or backbone feature
    target layer name of spatial attention mask:
        da_heads.semantic_list.0.4 Conv2d
        da_heads.semantic_list.1.4 Conv2d
        da_heads.semantic_list.2.4 Conv2d
        da_heads.semantic_list.3.4 Conv2d
        ..., the numer is the same as number of window
    target layer name of backbone feature:
        the last convolution layer of backbone
    '''
    def __init__(self, cfg, domain, target_feature, test_set):
        assert domain in ['source', 'target']
        assert target_feature in ['backbone', 'attention mask']
        self.target_feature = target_feature
        self.domain = domain
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        data_loader = DATrainer.build_test_loader(cfg, test_set)
        self.data_size = len(data_loader) # for `grad_cam_domain_classsifier` in train_net.py to track dataloader's size
        self.dataloader = iter(data_loader)
        self.handlers = []
        if target_feature == 'attention mask':
            self.target_layer_name_list = self.attention_mask_target_layer_name()
        else:
            self.target_layer_name_list = self.backbone_target_layer_name()
        self.features = [None] * len(self.target_layer_name_list)
        self.gradients = [None] * len(self.target_layer_name_list)
        self._register_hook()
        # source: https://stackoverflow.com/questions/57323023/pytorch-loss-backward-and-optimizer-step-in-eval-mode-with-batch-norm-laye
        # eval mode does not block parameters to be updated, 
        # it only changes the behaviour of some layers (batch norm and dropout) during the forward pass
        self.model.eval()

    def _get_features_hook(self, module, input, output, key=0):
        self.features[key] = output.detach()

    def _get_grads_hook(self, module, grad_in, grad_out, key=0):
        self.gradients[key] = grad_out[0].detach()

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def _register_hook(self):
        if self.target_feature == 'attention mask':
            target_module = self.model.da_heads
            for (name, module) in target_module.named_modules():
                if name in self.target_layer_name_list:
                    key = int(name.split('.')[-2])
                    self.handlers.append(module.register_forward_hook(partial(self._get_features_hook, key=key)))
                    self.handlers.append(module.register_full_backward_hook(partial(self._get_grads_hook, key=key)))

        else:
            target_module = self.model.backbone
            for (name, module) in target_module.named_modules():
                if name in self.target_layer_name_list:
                    self.handlers.append(module.register_forward_hook((self._get_features_hook)))
                    self.handlers.append(module.register_full_backward_hook(self._get_grads_hook))


    def attention_mask_target_layer_name(self):
        '''
        Return: list[str], taget layer name in domain classifier
        '''
        name_list = set()
        for name, m in self.model.da_heads.named_modules():
            if 'semantic_list' in name and name.split('.')[-1] == '4' and isinstance(m, torch.nn.Conv2d):
                name_list.add(name)
        return name_list

    def backbone_target_layer_name(self):
        '''
        Return: list[str], taget layer name in domain classifier
        '''
        layer_name = None
        for name, m in self.model.backbone.named_modules():
            if isinstance(m, torch.nn.Conv2d):
                layer_name = name
        return [layer_name]

    def __call__(self):
        data = next(self.dataloader) # batch size is 1
        _, im_h, im_w = data[0]['image'].size()
        self.model.zero_grad()
        domain_vector:torch.Tensor = self.model.get_domain_vector(data)
        domain_vector = domain_vector.softmax(dim=1)[0] # [N, 2], where N is batch size, is 1
        if self.domain == 'target':
            domain_vector[1].backward()
            prob = domain_vector[1]
        else:
            domain_vector[0].backward()
            prob = domain_vector[0]
        final_cam = 0.
        for g, f in zip(self.gradients, self.features):
            weight = g.mean(dim=(-2,-1), keepdim=True)
            cam:torch.Tensor = weight * f
            cam = cam.sum(dim=(0, 1)) # [H, W]
            cam = F.relu(cam) # [H, W]
            cam -= cam.min()
            if cam.max() != 0: cam /= cam.max()
            cam = F.interpolate(cam.view(1, 1, cam.size(0), cam.size(1)), size=(im_h, im_w), mode='bicubic', align_corners=True)
            cam = cam.view(cam.size(-2), cam.size(-1))
            cam = cam.clamp(min=0., max=1.)
            final_cam += cam
        final_cam = final_cam / len(self.target_layer_name_list)
        final_cam = final_cam / final_cam.max()
        return final_cam, data[0]['image'], data[0]['file_name'], prob


class GramCamForObjectDetection:
    '''
    Grad cam for object detector, target layer is the last convolution layer of roi_head
    '''
    def __init__(self, cfg, test_set):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.target_layer:str = self.target_layer_name()
        print(self.target_layer)
        data_loader = DATrainer.build_test_loader(cfg, test_set)
        self.data_size = len(data_loader) # for `grad_cam_object_detection` in train_net.py to track dataloader's size
        self.dataloader = iter(data_loader)
        self.handlers = []
        self.feature = None
        self.gradient = None
        self._register_hook()
        self.model.eval()
        self.sc = 1.2
        self.thr1, self.thr2 = 0.1, 0.02

    def _get_features_hook(self, module, input, output):
        self.feature = output.detach()

    def _get_grads_hook(self, module, grad_in, grad_out):
        self.gradient = grad_out[0].detach()

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def target_layer_name(self):
        '''
        Return: str, taget layer name in domain classifier
        '''
        layer_name = None
        for name, m in self.model.named_modules():
            if isinstance(m, torch.nn.Conv2d):
                layer_name = name
        return layer_name

    def _register_hook(self):
        for (name, module) in self.model.named_modules():
            if name == self.target_layer:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_full_backward_hook(self._get_grads_hook))
                break

    def __call__(self):
        data = next(self.dataloader) # batch size is 1
        im_h, im_w = data[0]['height'], data[0]['width']
        self.model.zero_grad()
        output = self.model.inference(data)
        output = output[0]['instances']
        final_cam = torch.zeros((im_h, im_w), device=next(self.model.parameters()).device)
        for score, box, ind in zip(output.scores, output.pred_boxes, output.indices):
            score = score * 2
            score.backward(retain_graph=True)
            gradient = self.gradient[ind]
            feature:torch.Tensor = self.feature[ind]
            weight = gradient.mean(dim=(1,2), keepdim=True)
            weight = weight.sum(dim=(1,2), keepdim=True)
            cam = F.relu((weight * feature).sum(dim=0))
            cam -= cam.min()
            ma = cam.max()
            if ma > 0:
                cam /= ma
            x1, y1, x2, y2 = box.detach()
            x1, y1, x2, y2 = x1.int(), y1.int(), x2.int(), y2.int()
            h, w = (y2-y1).item(), (x2-x1).item()
            gh, gw = self.sc*h, self.sc*w
            cam *= 255
            cam = F.interpolate(cam.view(1, 1, cam.size(0), cam.size(1)), size=(int(gh), int(gw)), mode='bicubic', align_corners=True)
            cam = cam.view(cam.size(-2), cam.size(-1)) / 255
            cam = centcp(cam, (h, w))
            cam = cam.clamp(min=0., max=1.)
            cam[cam<=self.thr1] *= cam[cam<=self.thr1]
            cam[cam<self.thr2] = 0. 
            final_cam[y1:y2, x1:x2] += cam
            self.model.zero_grad()
        final_cam = final_cam.clamp(min=0., max=1.)
        return final_cam, data[0]['file_name']
    
    
def process_pred2label(target_output, threshold=0.7):
    from .modeling.bounding_box import BoxList
    pseudo_labels_list = []
    masks = []
    output_instances = []
    for idx, bbox_l in enumerate(target_output):
        # get predict boxes
        pred_bboxes = bbox_l._fields['pred_boxes'].tensor.detach()
        # get predict logits(which can be negative and need to be converted to probability))
        scores = bbox_l._fields['scores'].detach()
        # set predict labels, only works for single class
        labels = torch.zeros(scores.shape, dtype=torch.int64).to(scores.device).detach()
        # filter out low probability boxes, and its corresponding labels
        filtered_idx = scores>=threshold
        filtered_bboxes = pred_bboxes[filtered_idx]
        filtered_labels = labels[filtered_idx]
        # create new BoxList, which is used to store filtered_bboxes(Tensor)
        new_bbox_list = BoxList(filtered_bboxes, bbox_l._image_size, mode="xyxy")
        # convert to gt_instances format(Instances)
        tmp = Instances(new_bbox_list.size)
        tmp.gt_boxes = Boxes(new_bbox_list.bbox)
        tmp.gt_classes = filtered_labels
        output_instances.append(tmp)
    return output_instances, masks

def my_preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images

def pseudo_dataset_gen(cfg, t_data, t_out):
    from pascal_voc_writer import Writer
    from detectron2.utils.visualizer import Visualizer, ColorMode
    from detectron2.data.datasets import load_voc_instances
    import cv2
    from datetime import datetime
    from pathlib import Path
    from detectron2.engine import DefaultPredictor

    for dataset_name in cfg.DATASETS.TRAIN:
        now = datetime.now()
        xml_path =  Path(__file__).parent.parent/'datasets'/'Cityscapes-coco'/'pseudo_img_v2'/'Annotations'
        image_path = Path(__file__).parent.parent/'datasets'/'Cityscapes-coco'/'pseudo_img_v2'/'JPEGImages'
        split_path = Path(__file__).parent.parent/'datasets'/'Cityscapes-coco'/'pseudo_img_v2'/'ImageSets'/'Main'
        xml_path.mkdir(parents=True, exist_ok=True)
        image_path.mkdir(parents=True, exist_ok=True)
        split_path.mkdir(parents=True, exist_ok=True)
        
        im1 = cv2.imread(t_data[0].get('file_name'), cv2.IMREAD_COLOR)
        im2 = cv2.imread(t_data[1].get('file_name'), cv2.IMREAD_COLOR)
        writer = Writer(t_data[1].get('file_name')[:-4] + '_pseudo' + '.jpg', 2048, 1024)
        # if t_data[1].get('image_id') == 'aachen_000067_000019_leftImg8bit' or t_data[0].get('image_id') == 'aachen_000067_000019_leftImg8bit':
        #     print(1)
        writed_box = []
        background_img_list = t_out[1]._fields['pred_boxes']
        add_img_list = t_out[0]._fields['pred_boxes']



        ##################V1##################
        # for i, box in enumerate(add_img_list):
        #     x1, y1, x2, y2 = map(int, box)
        #     x1, y1, x2, y2 = map(lambda x: 0 if x < 0 else x, [x1, y1, x2, y2])
        #     _trim = False
        #     for j, bbox in enumerate(background_img_list):
        #         if intersect_ratios(box, bbox)[0] >= 0.65 or bbox_inside(box, bbox) or bbox_inside(bbox, box):
        #             _trim = True
        #             break
        #     for wbox in writed_box:
        #         if intersect_ratios(box, wbox)[0] >= 0.65 or bbox_inside(box, wbox) or bbox_inside(wbox, box):
        #             _trim = True
        #             break
        #     if not _trim:
        #         im2[y1 : y2, x1 : x2] = im1[y1 : y2, x1 : x2]
        #         writer.addObject('car', x1, y1, x2, y2)
        #         writed_box.append(box)
        # for j, bbox in enumerate(background_img_list):
        #     x1, y1, x2, y2 = map(int, bbox)
        #     x1, y1, x2, y2 = map(lambda x: 0 if x < 0 else x, [x1, y1, x2, y2])
        #     for wbox in writed_box:
        #         if intersect_ratios(bbox, wbox)[0] >= 0.65 or bbox_inside(bbox, wbox) or bbox_inside(wbox, bbox):
        #             continue
        #         writer.addObject('car', x1, y1, x2, y2)
                    
        # xml_name = t_data[1].get('image_id') + '_pseudo' + '.xml'
        # writer.save(xml_path/xml_name)
        # with open(split_path/'train.txt', 'a') as f:
        #     f.write(t_data[1].get('image_id') + '_pseudo' + '\n')
        # v = Visualizer(im2[:, :, ::-1], MetadataCatalog.get(dataset_name), scale=1.0, instance_mode=ColorMode.IMAGE)
        # image_name = str(image_path/'{}.png').format(Path(t_data[1].get('file_name')).stem)
        # cv2.imwrite(image_name, v.img)
        # os.rename(image_name, image_name[:-4] + '_pseudo' + '.jpg')

        ##################V2##################
        # sort boxes by area
        tmp = []
        for i, box in enumerate(add_img_list):
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            tmp.append({'area': area, 'box': box})
        tmp.sort(key=lambda tmp: tmp['area'], reverse=True)
        sorted_boxes_big = []
        sorted_area_big = []
        sorted_boxes_small = []
        sorted_area_small = []
        for l in tmp:
            if l['area'] >= 100000:
                sorted_boxes_big.append(l['box'])
                sorted_area_big.append(l['area'])
            else:
                sorted_boxes_small.append(l['box'])
                sorted_area_small.append(l['area'])
        
        # do big
        for i, box in enumerate(sorted_boxes_big):
            x1, y1, x2, y2 = map(int, box)
            x1, y1, x2, y2 = map(lambda x: 0 if x < 0 else x, [x1, y1, x2, y2])
            _trim = False
            for j, box2 in enumerate(background_img_list):
                if intersect_ratios(box, box2)[0] >= 0.65 or bbox_inside(box, box2):
                    _trim = True
                    break
            for wbox in writed_box:
                if intersect_ratios(box, wbox)[0] >= 0.65 or bbox_inside(box, wbox):
                    _trim = True
                    break
            if not _trim:
                im2[y1 : y2, x1 : x2] = im1[y1 : y2, x1 : x2]
                writer.addObject('car', x1, y1, x2, y2)
                writed_box.append(box)
        for j, box2 in enumerate(background_img_list):
            x1, y1, x2, y2 = map(int, box2)
            x1, y1, x2, y2 = map(lambda x: 0 if x < 0 else x, [x1, y1, x2, y2])
            _trim = False
            for wbox in writed_box:
                if intersect_ratios(box2, wbox)[0] >= 0.65 or bbox_inside(box2, wbox):
                    _trim = True
                    break
            if not _trim:
                writer.addObject('car', x1, y1, x2, y2)

        xml_name = t_data[1].get('image_id') + '_big_pseudo' + '.xml'
        writer.save(xml_path/xml_name)
        with open(split_path/'train.txt', 'a') as f:
            f.write(t_data[1].get('image_id') + '_big_pseudo' + '\n')
        v = Visualizer(im2[:, :, ::-1], MetadataCatalog.get(dataset_name), scale=1.0, instance_mode=ColorMode.IMAGE)
        image_name = str(image_path/'{}_big.png').format(Path(t_data[1].get('file_name')).stem)
        v.get_output().save(image_name)
        os.rename(image_name, image_name[:-4] + '_pseudo' + '.jpg')

        # do small
        for i, box in enumerate(sorted_boxes_small):
            x1, y1, x2, y2 = map(int, box)
            x1, y1, x2, y2 = map(lambda x: 0 if x < 0 else x, [x1, y1, x2, y2])
            _trim = False
            for j, box2 in enumerate(background_img_list):
                if intersect_ratios(box, box2)[0] >= 0.65 or bbox_inside(box, box2):
                    _trim = True
                    break
            for wbox in writed_box:
                if intersect_ratios(box, wbox)[0] >= 0.65 or bbox_inside(box, wbox):
                    _trim = True
                    break
            if not _trim:
                im2[y1 : y2, x1 : x2] = im1[y1 : y2, x1 : x2]
                writer.addObject('car', x1, y1, x2, y2)
                writed_box.append(box)
        for j, box2 in enumerate(background_img_list):
            x1, y1, x2, y2 = map(int, box2)
            x1, y1, x2, y2 = map(lambda x: 0 if x < 0 else x, [x1, y1, x2, y2])
            _trim = False
            for wbox in writed_box:
                if intersect_ratios(box2, wbox)[0] >= 0.65 or bbox_inside(box2, wbox):
                    _trim = True
                    break
            if not _trim:
                writer.addObject('car', x1, y1, x2, y2)
                    
        xml_name = t_data[1].get('image_id') + '_small_pseudo' + '.xml'
        writer.save(xml_path/xml_name)
        with open(split_path/'train.txt', 'a') as f:
            f.write(t_data[1].get('image_id') + '_small_pseudo' + '\n')
        v = Visualizer(im2[:, :, ::-1], MetadataCatalog.get(dataset_name), scale=1.0, instance_mode=ColorMode.IMAGE)
        image_name = str(image_path/'{}_small.png').format(Path(t_data[1].get('file_name')).stem)
        v.get_output().save(image_name)
        os.rename(image_name, image_name[:-4] + '_pseudo' + '.jpg')

'''
# def location_aware_trim():
#     annotations_trimmed = []
#     for ann in t_out[0]._fields['pred_boxes']:
#         _trim = False
#         for ann2 in t_out[1]._fields['pred_boxes']:
#             if intersect_ratios(ann, ann2)[0] >= 0.65 or bbox_inside(ann, ann2):
#                 _trim = True
#                 break
#         if not _trim:
#             annotations_trimmed.append(ann)
#     for ann in t_out[1]._fields['pred_boxes']:
#         annotations_trimmed.append(ann)
#     return annotations_trimmed
'''
def intersect_ratios(bbox1, bbox2):
    x11, y11, x12, y12 = bbox1
    x21, y21, x22, y22 = bbox2
    xA, yA = max(x11,x21), max(y11,y21)
    xB, yB = min(x12,x22), min(y12,y22)

    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)
    overlap = max(xB - xA, 0) * max(yB - yA, 0)
    return overlap / area1, overlap / area2

def bbox_inside(bbox1, bbox2, eps=1.0):
    x11, y11, x12, y12 = bbox1
    x21, y21, x22, y22 = bbox2
    return x11 + eps >= x21 and y11 + eps >= y21 and x12 <= x22 + eps and y12 <= y22 + eps