# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import warnings

from lichee import config
from lichee import plugin
from lichee.module.torch.op.target_assigner import GridAssigner
from lichee.module.torch.op.target_sampler import PseudoSampler
from lichee.module.torch.op.bbox_coder import YOLOBBoxCoder
from lichee.module.torch.op.anchor_generator import YOLOAnchorGenerator
from lichee.module.torch.op.nms_ops import multiclass_nms
from lichee.module.torch.layer.det_conv_module import ConvModule
from lichee.utils.common import multi_apply


def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def images_to_levels(target, num_levels):
    """Convert targets by image to targets by feature level.
    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_levels:
        end = start + n
        # level_targets.append(target[:, start:end].squeeze(0))
        level_targets.append(target[:, start:end])
        start = end
    return level_targets


@plugin.register_plugin(plugin.PluginType.TASK, "yolo_head")
class YOLOV3Head(nn.Module):
    """YOLOV3Head Paper link: https://arxiv.org/abs/1804.02767."""

    def __init__(self, target_cfg=None):
        super(YOLOV3Head, self).__init__()

        self.cfg = config.get_cfg()

        self.num_classes = self.cfg.MODEL.TASK.CONFIG.NUM_CLASSES
        self.in_channels = self.cfg.MODEL.TASK.CONFIG.IN_CHANNELS
        self.out_channels = self.cfg.MODEL.TASK.CONFIG.OUT_CHANNELS
        self.featmap_strides = self.cfg.MODEL.TASK.CONFIG.FEATMAP_STRIDES
        self.norm_type = self.cfg.MODEL.TASK.CONFIG.NORM_TYPE
        self.act_type = self.cfg.MODEL.TASK.CONFIG.ACT_TYPE
        self.test_param = self.cfg.MODEL.TASK.CONFIG.TEST_PARAM
        
        # Check params
        assert (len(self.in_channels) == len(self.out_channels) == len(self.featmap_strides))

        self.assigner = GridAssigner(
            pos_iou_thr=self.cfg.MODEL.TASK.CONFIG.GRIDASSIGNER['POS_IOU_THR'],
            neg_iou_thr=self.cfg.MODEL.TASK.CONFIG.GRIDASSIGNER['NEG_IOU_THR'],
            min_pos_iou=self.cfg.MODEL.TASK.CONFIG.GRIDASSIGNER['MIN_POS_IOU'])
            
        self.sampler = PseudoSampler(context=self)
        self.bbox_coder = YOLOBBoxCoder(
            scale_x_y=self.cfg.MODEL.TASK.CONFIG.YOLO_BBOX_CODER['SCALE_X_Y'])
        
        self.anchor_generator = YOLOAnchorGenerator(
            strides=self.cfg.MODEL.TASK.CONFIG.YOLO_ANCHOR_GENERATOR['STRIDES'],
            base_sizes=self.cfg.MODEL.TASK.CONFIG.YOLO_ANCHOR_GENERATOR['BASE_SIZES']
        )

        # usually the numbers of anchors for each level are the same
        # except SSD detectors
        self.num_anchors = self.anchor_generator.num_base_anchors[0]
        assert len(self.anchor_generator.num_base_anchors) == len(self.featmap_strides)

        loss_cls = plugin.get_plugin(
            plugin.PluginType.MODULE_LOSS, 
            self.cfg.MODEL.TASK.CONFIG.LOSS_CLS['NAME'])
        self.loss_cls = loss_cls.build(self.cfg.MODEL.TASK.CONFIG.LOSS_CLS)

        loss_conf = plugin.get_plugin(
            plugin.PluginType.MODULE_LOSS, 
            self.cfg.MODEL.TASK.CONFIG.LOSS_CONF['NAME'])
        self.loss_conf = loss_conf.build(self.cfg.MODEL.TASK.CONFIG.LOSS_CONF)

        loss_xy = plugin.get_plugin(
            plugin.PluginType.MODULE_LOSS, 
            self.cfg.MODEL.TASK.CONFIG.LOSS_XY['NAME'])
        self.loss_xy = loss_xy.build(self.cfg.MODEL.TASK.CONFIG.LOSS_XY)
        
        loss_wh = plugin.get_plugin(
            plugin.PluginType.MODULE_LOSS, 
            self.cfg.MODEL.TASK.CONFIG.LOSS_WH['NAME'])
        self.loss_wh = loss_wh.build(self.cfg.MODEL.TASK.CONFIG.LOSS_WH)

        self._init_layers()

    @property
    def num_levels(self):
        return len(self.featmap_strides)

    @property
    def num_attrib(self):
        """int: number of attributes in pred_map, bboxes (4) +
        objectness (1) + num_classes"""

        return 5 + self.num_classes

    def _init_layers(self):
        self.convs_bridge = nn.ModuleList()
        self.convs_pred = nn.ModuleList()
        for i in range(self.num_levels):
            conv_bridge = ConvModule(
                self.in_channels[i],
                self.out_channels[i],
                3,
                padding=1,
                norm_type=self.norm_type,
                act_type=self.act_type)
            conv_pred = nn.Conv2d(self.out_channels[i],
                                  self.num_anchors * self.num_attrib, 1)

            self.convs_bridge.append(conv_bridge)
            self.convs_pred.append(conv_pred)

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.convs_pred:
            normal_init(m, std=0.01)

    def forward(self, feats, label_inputs):
        """Forward features from the upstream network.
        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            tuple[Tensor]: A tuple of multi-level predication map, each is a
                4D-tensor of shape (batch_size, 5+num_classes, height, width).
        """
        assert len(feats) == self.num_levels
        pred_maps = []
        for i in range(self.num_levels):
            x = feats[i]
            x = self.convs_bridge[i](x)
            pred_map = self.convs_pred[i](x)
            pred_maps.append(pred_map)
        
        out_maps = tuple(pred_maps)

        gt_bboxes = label_inputs.get("gt_bboxes", None)
        gt_labels = label_inputs.get("gt_labels", None)

        if (gt_bboxes is None) or (gt_labels is None):
            return self.get_bboxes(out_maps)
        else:
            loss_dict = self.loss(out_maps, gt_bboxes, gt_labels)
            log_vars = OrderedDict()
            for loss_name, loss_value in loss_dict.items():
                if isinstance(loss_value, torch.Tensor):
                    log_vars[loss_name] = loss_value.mean()
                elif isinstance(loss_value, list):
                    log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
                else:
                    raise TypeError(
                        f'{loss_name} is not a tensor or list of tensors')

            loss_all = sum(_value for _key, _value in log_vars.items()
                           if 'loss' in _key)
            return list(out_maps), loss_all

    def get_bboxes(self, pred_maps, cfg_dict=None, with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            pred_maps (list[Tensor]): Raw predictions for a batch of images.
            cfg_dict (Dict | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        result_list = []
        num_levels = len(pred_maps)
        num_imgs = len(pred_maps[0])
        for img_id in range(num_imgs):
            pred_maps_list = [
                pred_maps[i][img_id].detach() for i in range(num_levels)
            ]
            proposals = self._get_bboxes_single(pred_maps_list, cfg_dict, with_nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           pred_maps_list,
                           scale_factor,
                           cfg_dict,
                           with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.
        Args:
            pred_maps_list (list[Tensor]): Prediction maps for different scales
                of each single image in the batch.
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (Dict | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """

        test_cfg_dict = self.test_param if cfg_dict is None else cfg_dict
        assert len(pred_maps_list) == self.num_levels

        multi_lvl_bboxes = []
        multi_lvl_cls_scores = []
        multi_lvl_conf_scores = []
        num_levels = len(pred_maps_list)
        featmap_sizes = [
            pred_maps_list[i].shape[-2:] for i in range(num_levels)
        ]
        multi_lvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, pred_maps_list[0][0].device)

        for i in range(self.num_levels):
            # get some key info for current scale
            pred_map = pred_maps_list[i]
            stride = self.featmap_strides[i]

            # (h, w, num_anchors*num_attrib) -> (h*w*num_anchors, num_attrib)
            pred_map = pred_map.permute(1, 2, 0).reshape(-1, self.num_attrib)

            pred_map[..., :2] = torch.sigmoid(pred_map[..., :2])
            bbox_pred = self.bbox_coder.decode(multi_lvl_anchors[i],
                                               pred_map[..., :4], stride)
            # conf and cls
            conf_pred = torch.sigmoid(pred_map[..., 4]).view(-1)
            cls_pred = torch.sigmoid(pred_map[..., 5:]).view(
                -1, self.num_classes)  # Cls pred one-hot.
            
            # Filtering out all predictions with conf < conf_thr
            # Get top-k prediction
            if not torch.onnx.is_in_onnx_export():
                conf_thr = cfg.get('conf_thr', -1)
                conf_inds = conf_pred.ge(conf_thr).nonzero().flatten()
                bbox_pred = bbox_pred[conf_inds, :]
                cls_pred = cls_pred[conf_inds, :]
                conf_pred = conf_pred[conf_inds]
                
            nms_pre = cfg.get('nms_pre', -1)
            if 0 < nms_pre < conf_pred.size(0):
                _, topk_inds = torch.topk(conf_pred, nms_pre)
                bbox_pred = bbox_pred[topk_inds, :]
                cls_pred = cls_pred[topk_inds, :]
                conf_pred = conf_pred[topk_inds]

            # Save the result of current scale
            multi_lvl_bboxes.append(bbox_pred)
            multi_lvl_cls_scores.append(cls_pred)
            multi_lvl_conf_scores.append(conf_pred)
        
        # Merge the results of different scales together
        multi_lvl_bboxes = torch.cat(multi_lvl_bboxes)
        multi_lvl_cls_scores = torch.cat(multi_lvl_cls_scores)
        multi_lvl_conf_scores = torch.cat(multi_lvl_conf_scores)

        if with_nms and (multi_lvl_conf_scores.size(0) == 0):
            return torch.zeros((0, 5)), torch.zeros((0, ))

        # the class_id for background is num_classes. i.e., the last column.
        padding = multi_lvl_cls_scores.new_zeros(multi_lvl_cls_scores.shape[0], 1)
        multi_lvl_cls_scores = torch.cat([multi_lvl_cls_scores, padding], dim=1)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms(
                multi_lvl_bboxes,
                multi_lvl_cls_scores,
                test_cfg_dict['SCORE_THR'],
                test_cfg_dict['NMS_IOU_THRESHOLD'],
                test_cfg_dict['MAX_PER_IMG'],
                score_factors=multi_lvl_conf_scores)
            return det_bboxes, det_labels
        else:
            return (multi_lvl_bboxes, multi_lvl_cls_scores, multi_lvl_conf_scores)

    def loss(self, pred_maps, gt_bboxes, gt_labels):
        """Compute loss of the head.

        Args:
            pred_maps (list[Tensor]): Prediction map for each scale level,
                shape (N, num_anchors * num_attrib, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_imgs = len(gt_bboxes)
        device = pred_maps[0].device

        featmap_sizes = [
            pred_maps[i].shape[-2:] for i in range(self.num_levels)
        ]
        multi_level_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        responsible_flag_list = []
        for img_id in range(num_imgs):
            responsible_flag_list.append(
                self.anchor_generator.responsible_flags(
                    featmap_sizes, gt_bboxes[img_id], device))

        target_maps_list, neg_maps_list = self.get_targets(
            anchor_list, responsible_flag_list, gt_bboxes, gt_labels)

        losses_cls, losses_conf, losses_xy, losses_wh = multi_apply(
            self.loss_single, pred_maps, target_maps_list, neg_maps_list)

        return dict(
            loss_cls=losses_cls,
            loss_conf=losses_conf,
            loss_xy=losses_xy,
            loss_wh=losses_wh)

    def loss_single(self, pred_map, target_map, neg_map):
        """Compute loss of a single image from a batch.
        Args:
            pred_map (Tensor): Raw predictions for a single level.
            target_map (Tensor): The Ground-Truth target for a single level.
            neg_map (Tensor): The negative masks for a single level.
        Returns:
            tuple:
                loss_cls (Tensor): Classification loss.
                loss_conf (Tensor): Confidence loss.
                loss_xy (Tensor): Regression loss of x, y coordinate.
                loss_wh (Tensor): Regression loss of w, h coordinate.
        """

        num_imgs = len(pred_map)
        pred_map = pred_map.permute(0, 2, 3,
                                    1).reshape(num_imgs, -1, self.num_attrib)
        neg_mask = neg_map.float()
        pos_mask = target_map[..., 4]
        pos_and_neg_mask = neg_mask + pos_mask
        pos_mask = pos_mask.unsqueeze(dim=-1)
        if torch.max(pos_and_neg_mask) > 1.:
            warnings.warn('There is overlap between pos and neg sample.')
            pos_and_neg_mask = pos_and_neg_mask.clamp(min=0., max=1.)

        pred_xy = pred_map[..., :2]
        pred_wh = pred_map[..., 2:4]
        pred_conf = pred_map[..., 4]
        pred_label = pred_map[..., 5:]

        target_xy = target_map[..., :2]
        target_wh = target_map[..., 2:4]
        target_conf = target_map[..., 4]
        target_label = target_map[..., 5:]

        loss_cls = self.loss_cls(pred_label, target_label, weight=pos_mask)
        loss_conf = self.loss_conf(
            pred_conf, target_conf, weight=pos_and_neg_mask)
        loss_xy = self.loss_xy(pred_xy, target_xy, weight=pos_mask)
        loss_wh = self.loss_wh(pred_wh, target_wh, weight=pos_mask)

        return loss_cls, loss_conf, loss_xy, loss_wh

    def get_targets(self, anchor_list, responsible_flag_list, gt_bboxes_list,
                    gt_labels_list):
        """Compute target maps for anchors in multiple images.
        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_total_anchors, 4).
            responsible_flag_list (list[list[Tensor]]): Multi level responsible
                flags of each image. Each element is a tensor of shape
                (num_total_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
        Returns:
            tuple: Usually returns a tuple containing learning targets.
                - target_map_list (list[Tensor]): Target map of each level.
                - neg_map_list (list[Tensor]): Negative map of each level.
        """
        num_imgs = len(anchor_list)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

        results = multi_apply(self._get_targets_single, anchor_list,
                              responsible_flag_list, gt_bboxes_list,
                              gt_labels_list)

        all_target_maps, all_neg_maps = results
        assert num_imgs == len(all_target_maps) == len(all_neg_maps)
        target_maps_list = images_to_levels(all_target_maps, num_level_anchors)
        neg_maps_list = images_to_levels(all_neg_maps, num_level_anchors)

        return target_maps_list, neg_maps_list

    def _get_targets_single(self, anchors, responsible_flags, gt_bboxes,
                            gt_labels):
        """Generate matching bounding box prior and converted GT.
        Args:
            anchors (list[Tensor]): Multi-level anchors of the image.
            responsible_flags (list[Tensor]): Multi-level responsible flags of
                anchors
            gt_bboxes (Tensor): Ground truth bboxes of single image.
            gt_labels (Tensor): Ground truth labels of single image.
        Returns:
            tuple:
                target_map (Tensor): Predication target map of each
                    scale level, shape (num_total_anchors,
                    5+num_classes)
                neg_map (Tensor): Negative map of each scale level,
                    shape (num_total_anchors,)
        """

        anchor_strides = []
        for i in range(len(anchors)):
            anchor_strides.append(
                torch.tensor(self.featmap_strides[i],
                             device=gt_bboxes.device).repeat(len(anchors[i])))
        concat_anchors = torch.cat(anchors)
        concat_responsible_flags = torch.cat(responsible_flags)

        anchor_strides = torch.cat(anchor_strides)
        assert len(anchor_strides) == len(concat_anchors) == \
               len(concat_responsible_flags)
        assign_result = self.assigner.assign(concat_anchors,
                                             concat_responsible_flags,
                                             gt_bboxes)
        sampling_result = self.sampler.sample(assign_result, concat_anchors,
                                              gt_bboxes)

        target_map = concat_anchors.new_zeros(
            concat_anchors.size(0), self.num_attrib)

        target_map[sampling_result.pos_inds, :4] = self.bbox_coder.encode(
            sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes.float(),
            anchor_strides[sampling_result.pos_inds])

        target_map[sampling_result.pos_inds, 4] = 1

        gt_labels_one_hot = F.one_hot(
            gt_labels, num_classes=self.num_classes).float()
        
        target_map[sampling_result.pos_inds, 5:] = gt_labels_one_hot[
            sampling_result.pos_assigned_gt_inds]

        neg_map = concat_anchors.new_zeros(
            concat_anchors.size(0), dtype=torch.uint8)
        neg_map[sampling_result.neg_inds] = 1

        return target_map, neg_map
