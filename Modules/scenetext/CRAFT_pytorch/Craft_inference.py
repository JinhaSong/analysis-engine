import sys
import os
import time
import argparse

import torch
import torch.backends.cudnn as cudnn

import cv2
import numpy as np
import pandas as pd
import shutil

# import file_utils
# import new_test
# from craft import CRAFT
from Modules.scenetext import file_utils
from Modules.scenetext.craft import CRAFT
from Modules.scenetext import new_test


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

trained_model = '/workspace/Modules/scenetext/weights/craft_mlt_25k.pth'
text_threshold = 0.8
low_text = 0.4
link_threshold = 0.4
cuda = str2bool
canvas_size = 1280
mag_ratio = 1.5
poly = True
show_time = False
refine = False
refiner_model = '/workspace/Modules/scenetext/weights/craft_refiner_CTW1500.pth'


def crop(pts, image):
  rect = cv2.boundingRect(pts)
  x,y,w,h = rect
  cropped = image[y:y+h, x:x+w].copy()
  pts = pts - pts.min(axis=0)
  mask = np.zeros(cropped.shape[:2], np.uint8)
  cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
  dst = cv2.bitwise_and(cropped, cropped, mask=mask)
  bg = np.ones_like(cropped, np.uint8)*255
  cv2.bitwise_not(bg,bg, mask=mask)
  dst2 = bg + dst

  return dst2


def generate_words(image_name, score_bbox, image, base_dir, imwrite = True):

  crop_images = []

  num_bboxes = len(score_bbox)
  #CHANGE DIR
  dir = base_dir

  if imwrite == True :
    shutil.rmtree(dir)
    os.mkdir(dir)

  for num in range(num_bboxes):
    bbox_coords = score_bbox[num].split(':')[-1].split(',\n')
    if bbox_coords!=['{}']:
      l_t = float(bbox_coords[0].strip(' array([').strip(']').split(',')[0])
      if l_t < 0 : # index started in negative value.... it makes the order of crop images tangle
        l_t = 0
      t_l = float(bbox_coords[0].strip(' array([').strip(']').split(',')[1])
      r_t = float(bbox_coords[1].strip(' [').strip(']').split(',')[0])
      t_r = float(bbox_coords[1].strip(' [').strip(']').split(',')[1])
      r_b = float(bbox_coords[2].strip(' [').strip(']').split(',')[0])
      b_r = float(bbox_coords[2].strip(' [').strip(']').split(',')[1])
      l_b = float(bbox_coords[3].strip(' [').strip(']').split(',')[0])
      b_l = float(bbox_coords[3].strip(' [').strip(']').split(',')[1].strip(']'))
      pts = np.array([[int(l_t), int(t_l)], [int(r_t) ,int(t_r)], [int(r_b) , int(b_r)], [int(l_b), int(b_l)]])
      if np.all(pts) >= -1:
        word = crop(pts, image)

        folder = '/'.join( image_name.split('/')[:-1])

        # if os.path.isdir(os.path.join(dir + folder)) == False :
        #   os.makedirs(os.path.join(dir + folder))
        crop_images.append(word)
        if imwrite == True :
            file_name = os.path.join(dir, image_name + '_{}_{}_{}_{}_{}_{}_{}_{}.jpg'.format(l_t, t_l, r_t ,t_r, r_b , b_r ,l_b, b_l))
            try :
                cv2.imwrite(file_name, word)
            except:
                pass

  return crop_images

def image_crop(image, data, base_dir):
    csv_path = os.path.join(base_dir, 'data.csv')
    data=pd.read_csv(csv_path)

    for image_num in range(data.shape[0]):
        image_name = data['image_name'][image_num].strip('.jpg')
        score_bbox = data['word_bboxes'][image_num].split('),')
        crop_images = generate_words(image_name, score_bbox, image, base_dir)

    return crop_images


image_names = []
image_paths = []

def inference_one_image(net, refine_net, image, current_frame, base_dir):
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    csv_path = os.path.join(base_dir, 'data.csv')
    data=pd.DataFrame(columns=['image_name', 'word_bboxes', 'pred_words', 'align_text'])
    data['image_name'] = [str(current_frame)+".jpg"]

    t = time.time()

    bboxes, polys, score_text, det_scores = new_test.test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, canvas_size, mag_ratio, refine_net)
    bbox_score={}

    for box_num in range(len(bboxes)):
      key = str(det_scores[box_num])
      item = bboxes[box_num]
      bbox_score[key]=item

    data['word_bboxes'][0]=bbox_score

    mask_file = base_dir + "/res_" + str(current_frame) + '_mask.jpg'  # heatmap , saveResult()에서 overriding 됨.
    cv2.imwrite(mask_file, score_text)
    detect_box_img = file_utils.saveResult(int(current_frame), image[:,:,::-1], polys, dirname=base_dir, imwrite = True)
    data.to_csv(csv_path, sep = ',', na_rep='Unknown')

    return bboxes, polys, score_text, det_scores, data, detect_box_img


def load_craft_weights() :
    net = CRAFT()

    if cuda:
        net.load_state_dict(new_test.copyStateDict(torch.load(trained_model)))
    else:
        net.load_state_dict(new_test.copyStateDict(torch.load(trained_model, map_location='cpu')))

    if cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    net.eval()

    # LinkRefiner
    refine_net = None
    if refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        if cuda:
            refine_net.load_state_dict(new_test.copyStateDict(torch.load(refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(new_test.copyStateDict(torch.load(refiner_model, map_location='cpu')))

        refine_net.eval()
        poly = True
    return net , refine_net

def craft_one_image(net, refine_net, img, current_frame, base_dir):
    bboxes, polys, score_text, det_scores, data, detect_box_img = inference_one_image(net, refine_net, img, current_frame, base_dir)
    crop_images = image_crop(img, data, base_dir)

    return bboxes, crop_images, detect_box_img