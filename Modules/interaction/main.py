import os
import time

import cv2

import argparse
import torch
import yaml
import pickle
import numpy as np

from AnalysisEngine import settings
from Modules.dummy.main import Dummy
from WebAnalyzer.utils.media import frames_to_timecode
from Modules.interaction.backbone import EfficientDetBackbone
from Modules.interaction.efficientdet.utils import BBoxTransform, ClipBoxes
from Modules.interaction.efficientdet.help_function import single_iou, single_ioa, single_inter, single_union, transform_action, label_to_class, sub_label_to_class
from Modules.interaction.utils.utils import preprocess, invert_affine, postprocess, postprocess_hoi, postprocess_dense_union, postprocess_hoi_flip, postprocess_dense_union_flip
from Modules.interaction.utils.apply_prior import apply_prior
from utils import Logging

class HOI(Dummy):
    model = None
    result = None
    path = os.path.dirname(os.path.abspath(__file__))

    def calc_ioa(self, a, b):
        # a(anchor) [boxes, (x1, y1, x2, y2)]
        # b(gt, coco-style) [boxes, (x1, y1, x2, y2)]

        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

        exp_x1 = np.expand_dims(a[:, 0], axis=1)
        exp_x2 = np.expand_dims(a[:, 2], axis=1)
        exp_y1 = np.expand_dims(a[:, 1], 1)
        exp_y2 = np.expand_dims(a[:, 3], 1)

        iw = np.where(exp_x2 < b[:, 2], exp_x2, b[:, 2]) - np.where(exp_x1 > b[:, 0], exp_x1, b[:, 0])
        ih = np.where(exp_y2 < b[:, 3], exp_y2, b[:, 3]) - np.where(exp_y1 > b[:, 1], exp_y1, b[:, 1])
        iw = np.where(iw > 0, iw, 0)
        ih = np.where(ih > 0, ih, 0)

        intersection = iw * ih
        area = np.where(area > 1e-6, area, 1e-6)
        IoA = intersection / area
        return IoA

    def calc_iou(self, a, b):
        # a(anchor) [boxes, (x1, y1, x2, y2)]
        # b(gt, coco-style) [boxes, (x1, y1, x2, y2)]

        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

        exp_x1 = np.expand_dims(a[:, 0], axis=1)
        exp_x2 = np.expand_dims(a[:, 2], axis=1)
        exp_y1 = np.expand_dims(a[:, 1], 1)
        exp_y2 = np.expand_dims(a[:, 3], 1)

        iw = np.where(exp_x2 < b[:, 2], exp_x2, b[:, 2]) - np.where(exp_x1 > b[:, 0], exp_x1, b[:, 0])
        ih = np.where(exp_y2 < b[:, 3], exp_y2, b[:, 3]) - np.where(exp_y1 > b[:, 1], exp_y1, b[:, 1])

        iw = np.where(iw > 0, iw, 0)
        ih = np.where(ih > 0, ih, 0)

        ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih
        ua = np.where(ua > 0, ua, 1e-8)

        intersection = iw * ih
        IoU = intersection / ua
        return IoU

    def xy_to_wh(self, bbox):
        ctr_x = (bbox[0] + bbox[2]) / 2
        ctr_y = (bbox[1] + bbox[3]) / 2
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return ctr_x, ctr_y, width, height

    def fetch_location_score(self, anchor_bbox, obj_bbox, target_bbox, human_bbox, sigma):
        xo, yo, wo, ho = self.xy_to_wh(obj_bbox)
        xt, yt, wt, ht = self.xy_to_wh(target_bbox)
        xa, ya, wa, ha = self.xy_to_wh(anchor_bbox)
        dist = np.zeros(2, dtype=np.float)
        dist[0] = (xo - xt) / wa
        dist[1] = (yo - yt) / ha

        return np.exp(-1 * np.sum(dist ** 2) / (2 * sigma ** 2))

    def target_object_dist(self, target_objects_pos, objects_pos, anchors):
        width = anchors[:, 2] - anchors[:, 0]
        height = anchors[:, 3] - anchors[:, 1]
        anchors_size = np.stack([width, height], axis=1)
        anchors_size = np.expand_dims(anchors_size, axis=1)
        target_objects_pos = np.expand_dims(target_objects_pos, 1)
        diff = target_objects_pos - objects_pos
        diff = diff / anchors_size
        dist = np.sum(diff ** 2, axis=2)
        return dist

    def hoi_match(self, preds_inst, preds_union, prior_mask, thre=0.05, human_thre=0.6, anchor_thre=0.25, loc_thre=0.1):
        num_inst = len(preds_inst["rois"])
        humans = []
        objects = []
        human_bboxes = []
        human_inst_ids = []
        human_role_scores = []

        while len(humans) == 0:
            if human_thre < 0.5:
                break
            for inst_id in range(num_inst):
                if preds_inst["obj_class_ids"][inst_id] != 0 or preds_inst["obj_scores"][inst_id] < human_thre:
                    continue
                item = {}
                item["bbox"] = preds_inst["rois"][inst_id]
                item["agent_scores"] = preds_inst["act_scores"][inst_id]
                item["role_scores"] = transform_action(preds_inst["act_scores"][inst_id], "subject")
                item["obj_scores"] = preds_inst["obj_scores"][inst_id]
                item["inst_id"] = inst_id
                humans.append(item)
                human_bboxes.append(item["bbox"])
                human_inst_ids.append(item["inst_id"])
                human_role_scores.append(item["role_scores"])
            human_thre -= 0.1
        human_bboxes = np.array(human_bboxes)
        human_inst_ids = np.array(human_inst_ids)
        human_role_scores = np.array(human_role_scores)

        obj_role_scores = []
        for obj_id in range(len(preds_inst["rois"])):
            item = {}
            obj_role_score = transform_action(preds_inst["act_scores"][obj_id], "object")
            obj_role_score = apply_prior(obj_role_score, preds_inst["obj_class_ids"][obj_id])
            item["obj_role_scores"] = obj_role_score

            item["obj_scores"] = preds_inst["obj_scores"][obj_id]

            item["obj_class_id"] = preds_inst["obj_class_ids"][obj_id]
            item["inst_id"] = obj_id
            obj_bbox = preds_inst["rois"][obj_id]
            item["bbox"] = obj_bbox
            objects.append(item)
            obj_role_scores.append(obj_role_score)
        object_bboxes = np.array(preds_inst["rois"])
        obj_role_scores = np.array(obj_role_scores)

        hoi_pair_score = np.zeros((len(humans), len(preds_inst["obj_class_ids"]), len(self.union_action_list)),
                                  dtype=np.float)
        if len(human_bboxes) > 0:
            IoA = self.calc_ioa(preds_union["rois"], human_bboxes)

            IoA_max = np.max(IoA, axis=1)
            human_foreground = IoA_max > 0.25
            human_IoA = IoA[human_foreground]
            for key in preds_union:
                preds_union[key] = preds_union[key][human_foreground]

            new_IoA = self.calc_ioa(preds_union["rois"], preds_inst["rois"])
            new_IoA_argmax = np.argmax(new_IoA, axis=1)
            new_IoA[np.arange(new_IoA.shape[0]), new_IoA_argmax] = 0
            new_IoA_sec_max = np.max(new_IoA, axis=1)
            obj_foreground = new_IoA_sec_max > 0.25
            for key in preds_union:
                preds_union[key] = preds_union[key][obj_foreground]

            human_IoU = self.calc_iou(preds_union["rois"], human_bboxes)
            human_IoA = human_IoA[obj_foreground]
            human_IoU_argmax = np.argmax(human_IoU * (human_IoA > 0.25), axis=1)
            obj_IoA = self.calc_ioa(preds_union["rois"], preds_inst["rois"])

            num_union = len(preds_union["rois"])
            num_human = len(human_bboxes)

            sp_vectors = preds_union["sp_vector"]
            inter_human_regions = human_bboxes[human_IoU_argmax]
            humans_pos_x = (inter_human_regions[:, 0] + inter_human_regions[:, 2]) / 2
            humans_pos_y = (inter_human_regions[:, 1] + inter_human_regions[:, 3]) / 2
            humans_pos = np.stack([humans_pos_x, humans_pos_y], axis=1)
            inter_objects_pos = humans_pos + sp_vectors

            objects_pos_x = (object_bboxes[:, 0] + object_bboxes[:, 2]) / 2
            objects_pos_y = (object_bboxes[:, 1] + object_bboxes[:, 3]) / 2
            objects_pos = np.stack([objects_pos_x, objects_pos_y], axis=1)

            obj_dists = self.target_object_dist(inter_objects_pos, objects_pos, preds_union["rois"])
            inter_human_instids = human_inst_ids[human_IoU_argmax]
            obj_dists[np.arange(num_union), inter_human_instids] = 100
            obj_dists[obj_IoA < 0.25] = 100
            inter_obj_ids = np.argmin(obj_dists, 1)
            inter_obj_dist = obj_dists[np.arange(num_union), inter_obj_ids]

            sigma = 0.9
            location_scores = np.exp(-1 * inter_obj_dist / (2 * sigma ** 2))
            location_scores = np.where(location_scores < loc_thre, 0, location_scores)
            anchor_scores = preds_union["act_scores"]
            anchor_scores = np.where(anchor_scores < anchor_thre, 0, anchor_scores)

            inter_human_ids = human_IoU_argmax
            inter_human_role_score = human_role_scores[inter_human_ids]
            inst_object_role_score = obj_role_scores[inter_obj_ids]

            inter_scores = 0.5 * ((inter_human_role_score + inst_object_role_score) * anchor_scores).T * location_scores

            inter_scores = inter_scores.T
            inter_scores[inst_object_role_score == 0] = 0

            for human_id in range(num_human):
                human_inter = inter_human_ids == human_id
                human_inter_obj_id = inter_obj_ids[human_inter]
                human_inter_score = inter_scores[human_inter]

                for obj_id in range(num_inst):
                    hoi_pair_score[human_id, obj_id] = np.sum(human_inter_score[human_inter_obj_id == obj_id], axis=0)

        for human_id in range(len(humans)):
            hoi_pair_score[human_id, :, :] *= humans[human_id]["obj_scores"]

        for obj_id in range(len(preds_inst["rois"])):
            obj_cat = preds_inst["obj_class_ids"][obj_id]
            hoi_pair_score[:, obj_id, :] *= preds_inst["obj_scores"][obj_id]
            hoi_pair_score[:, obj_id, :] *= prior_mask[:, obj_cat + 1]

        if self.flip_test:
            hoi_pair_score /= 2

        dets = []
        for human_id, human in enumerate(humans):
            det = {}
            det["person_box"] = human["bbox"]

            max_score = np.max(hoi_pair_score[human_id], axis=0)
            max_obj_id = np.argmax(hoi_pair_score[human_id], axis=0)
            for id in range(len(label_to_class)):
                action = label_to_class[id][0]
                role = label_to_class[id][1]

                obj_bbox = preds_inst["rois"][max_obj_id[id]]
                if max_score[id] > thre or prior_mask[id, 0] < 0.1:
                    det[action + "_" + role] = (obj_bbox[0], obj_bbox[1], obj_bbox[2], obj_bbox[3], max_score[id])
                    agent_score = max_score[id]
                else:
                    if human["role_scores"][id] > 0.0 and prior_mask[id, 0] > 0.1:
                        det[action + "_" + role] = (
                        0, 0, 0, 0, human["role_scores"][id] * human["obj_scores"] * prior_mask[id, 0])
                        agent_score = human["role_scores"][id] * human["obj_scores"]

                    else:
                        det[action + "_" + role] = (0, 0, 0, 0, 0)
                        agent_score = 0

                if action + "_agent" not in det:
                    det[action + "_agent"] = agent_score
                else:
                    det[action + "_agent"] = max(agent_score, det[action + "_agent"])
            for i in range(len(sub_label_to_class)):
                action = sub_label_to_class[i]
                if action + "_agent" not in det:
                    det[action + "_agent"] = human["agent_scores"][i]
            dets.append(det)
        return dets

    def img_detect(self, file, model, input_size, regressBoxes, clipBoxes, prior_mask, threshold):
        img_path = file

        ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)
        if self.use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

        x = x.to(torch.float32 if not self.use_float16 else torch.float16).permute(0, 3, 1, 2)

        if self.flip_test:
            ids = torch.arange(x.shape[-1] - 1, -1, -1).long().cuda()
            x_flip = x[..., ids]
            x_cat = torch.cat([x, x_flip], 0)

        with torch.no_grad():
            if self.flip_test:

                features, union_act_cls, union_sub_reg, union_obj_reg, \
                inst_act_cls, inst_obj_cls, inst_bbox_reg, anchors = model(x_cat)

                anchors = torch.cat([anchors, anchors], 0)
                preds_union = postprocess_dense_union_flip(x_cat, anchors, union_act_cls, union_sub_reg, union_obj_reg,
                                                           regressBoxes, clipBoxes, 0.5, 1)
                preds_inst = postprocess_hoi_flip(x_cat, anchors, inst_bbox_reg, inst_obj_cls, inst_act_cls,
                                                  regressBoxes, clipBoxes, threshold, self.nms_threshold,
                                                  mode="object", classwise=True)
            else:

                features, union_act_cls, union_sub_reg, union_obj_reg, \
                inst_act_cls, inst_obj_cls, inst_bbox_reg, anchors = model(x)

                preds_union = postprocess_dense_union(x, anchors, union_act_cls, union_sub_reg, union_obj_reg,
                                                      regressBoxes, clipBoxes, 0.5, 1, classwise=True)
                preds_inst = postprocess_hoi(x, anchors, inst_bbox_reg, inst_obj_cls, inst_act_cls,
                                             regressBoxes, clipBoxes, threshold, self.nms_threshold,
                                             mode="object", classwise=True)
            preds_inst = invert_affine(framed_metas, preds_inst)[0]
            preds_union = invert_affine(framed_metas, preds_union)[0]
            dets = self.hoi_match(preds_inst, preds_union, prior_mask)

        return dets

    def __init__(self):
        super().__init__()
        start_time = time.time()
        self.compound_coef = 3
        self.nms_threshold = 0.3
        self.use_cuda = 1
        self.gpu = 0
        self.use_float16 = 0
        self.need_visual = True
        self.project = "vcoco"
        self.flip_test = 1

        self.params = yaml.safe_load(open(os.path.join(self.path, 'weights/vcoco.yml')))
        self.SET_NAME = self.params['val_set']
        self.project_name = self.params["project_name"]
        self.obj_list = self.params['obj_list']
        self.union_action_list = eval(self.params['union_action_list'])

        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        self.input_size = self.input_sizes[self.compound_coef]


        with open(os.path.join(self.path, "weights/new_prior_mask.pkl"), "rb") as file:
            self.prior_mask = pickle.load(file, encoding="bytes")

        self.model = EfficientDetBackbone(num_classes=len(eval(self.params["obj_list"])), num_union_classes=25,
                                     num_inst_classes=51, compound_coef=self.compound_coef,
                                     ratios=eval(self.params["anchors_ratios"]), scales=eval(self.params["anchors_scales"]))
        self.model.load_state_dict(torch.load(os.path.join(self.path, "weights/vcoco_best.pth"), map_location=torch.device('cpu')))
        self.model.requires_grad_(False)
        self.model.eval()

        if self.use_cuda:
            self.model = self.model.cuda()
        if self.use_float16:
            self.model = self.model.half()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        end_time = time.time()
        print(Logging.i("Model is successfully loaded - {} sec".format(end_time - start_time)))

    def inference_by_image(self, image_path, threshold=0.5):

        img_detection = self.img_detect(image_path, self.model, self.input_size, self.regressBoxes, self.clipBoxes, self.prior_mask, threshold=threshold)
        HO_dic = {}
        HO_set = set()
        count = 0
        frame_result = []
        for element in img_detection:
            action_count = -1
            person = element['person_box']
            person_x = person[0]
            person_y = person[1]
            person_w = person[2] - person[0]
            person_h = person[3] - person[1]
            person_result = {
                "label": [{"description": "person"}],
                "position": {"x": int(person_x), "y": int(person_y), "w": int(person_w), "h": int(person_h)},
                "interaction": []
            }
            for action_key, action_value in element.items():
                if (action_key.split('_')[-1] != 'agent') and action_key != 'image_id' and action_key != 'person_box':
                    if (not np.isnan(action_value[0])) and (action_value[4] > 0.2):
                        O_box = action_value[:4]
                        H_box = element['person_box']
                        action_count += 1
                        if tuple(O_box) not in HO_set:
                            HO_dic[tuple(O_box)] = count
                            HO_set.add(tuple(O_box))
                            count += 1
                        if tuple(H_box) not in HO_set:
                            HO_dic[tuple(H_box)] = count
                            HO_set.add(tuple(H_box))
                            count += 1
                        action_name = action_key.split('_')[0]
                        action_score = action_value[4]
                        action = {
                            "label": [
                                {
                                    "description": action_name,
                                    "score": action_score
                                }
                            ],
                            "position": {
                                "x": int(O_box[0]),
                                "y": int(O_box[1]),
                                "w": int(O_box[2] - O_box[0]),
                                "h": int(O_box[3] - O_box[1])
                            }
                        }
                        person_result["interaction"].append(action)
            frame_result.append(person_result)

        result = {"frame_result": frame_result}

        return result

    def inference_by_video(self, frame_path_list, infos):
        video_info = infos['video_info']
        frame_urls = infos['frame_urls']
        fps = video_info['extract_fps']
        print(Logging.i("Start inference by video"))
        results = {
            "model_name": "object_detection",
            "analysis_time": 0,
            "frame_results": []
        }

        start_time = time.time()
        for idx, (frame_path, frame_url) in enumerate(zip(frame_path_list, frame_urls)):
            if idx % 10 == 0:
                print(Logging.i("Processing... (index: {}/{} / frame number: {} / path: {})".format(idx, len(frame_path_list), int((idx + 1) * fps), frame_path)))
            result = self.inference_by_image(frame_path)
            result["frame_url"] = settings.MEDIA_URL + frame_url[1:]
            result["frame_number"] = int((idx + 1) * fps)
            result["timestamp"] = frames_to_timecode((idx + 1) * fps, fps)
            results["frame_results"].append(result)

        results["sequence_results"] = self.merge_sequence(results["frame_results"])

        end_time = time.time()
        results['analysis_time'] = end_time - start_time
        print(Logging.i("Processing time: {}".format(results['analysis_time'])))

        self.result = results

        return self.result

    def merge_sequence(self, result):
        sequence_results = []
        # TODO
        # - return format
        # [
        #     {
        #         "label": {
        #             "description": "class name",
        #             "score": 100 # 추가여부는 선택사항
        #         },
        #         "position": { # 추가여부는 선택사항
        #             "x": 100,
        #             "y": 100,
        #             "w": 100,
        #             "h": 100
        #         },
        #         "start_frame": 30,
        #         "end_frame": 300
        #     }
        #     ...
        # ]

        return sequence_results