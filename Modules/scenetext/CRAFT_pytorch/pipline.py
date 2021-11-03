import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import new_test
import imgproc
import file_utils
import json
import zipfile
import pandas as pd
import natsort
# import crop_images

from craft import CRAFT

from collections import OrderedDict

# from google.colab.patches import cv2_imshow

VIDEO_PATH = "./CRAFT_pytorch/videos/"

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def VIDEO_FRAME(VIDEO_PATH,FPS=1):
    video_path_list = os.listdir(VIDEO_PATH)

    for video_path in video_path_list:
        print(video_path)

        cap = cv2.VideoCapture(VIDEO_PATH + video_path)
        current_frame = 0
        VIDEO_FPS = cap.get(cv2.CAP_PROP_FPS)
        print("VIDEO_FPS =",VIDEO_FPS)

        while True:
            ret,Frame = cap.read()

            if ret == False : 
                break
            
            cv2.imwrite("./CRAFT_pytorch/video_frame/"+"%d.jpg"%current_frame,Frame)
            current_frame = current_frame + 1
            print(current_frame)

            if current_frame == 51:
                  break

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


def generate_words(image_name, score_bbox, image):

  num_bboxes = len(score_bbox)
  for num in range(num_bboxes):
    bbox_coords = score_bbox[num].split(':')[-1].split(',\n')
    if bbox_coords!=['{}']:
      l_t = float(bbox_coords[0].strip(' array([').strip(']').split(',')[0])
      t_l = float(bbox_coords[0].strip(' array([').strip(']').split(',')[1])
      r_t = float(bbox_coords[1].strip(' [').strip(']').split(',')[0])
      t_r = float(bbox_coords[1].strip(' [').strip(']').split(',')[1])
      r_b = float(bbox_coords[2].strip(' [').strip(']').split(',')[0])
      b_r = float(bbox_coords[2].strip(' [').strip(']').split(',')[1])
      l_b = float(bbox_coords[3].strip(' [').strip(']').split(',')[0])
      b_l = float(bbox_coords[3].strip(' [').strip(']').split(',')[1].strip(']'))
      pts = np.array([[int(l_t), int(t_l)], [int(r_t) ,int(t_r)], [int(r_b) , int(b_r)], [int(l_b), int(b_l)]])
      print(pts)
      if np.all(pts) > 0:
        
        word = crop(pts, image)
        
        folder = '/'.join( image_name.split('/')[:-1])

        #CHANGE DIR
        dir = './CRAFT_pytorch/result/crop_result/'

        # if os.path.isdir(os.path.join(dir + folder)) == False :
        #   os.makedirs(os.path.join(dir + folder))

        try:
          file_name = os.path.join(dir + image_name)
          cv2.imwrite(file_name+'_{}_{}_{}_{}_{}_{}_{}_{}.jpg'.format(l_t, t_l, r_t ,t_r, r_b , b_r ,l_b, b_l), word)
          print('Image saved to '+file_name+'_{}_{}_{}_{}_{}_{}_{}_{}.jpg'.format(l_t, t_l, r_t ,t_r, r_b , b_r ,l_b, b_l))
        except:
          continue


def image_crop():
  # pipline.video_info_csv()
  data=pd.read_csv('./CRAFT_pytorch/result/csv_result/data.csv')
  start = './CRAFT_pytorch/video_frame/'
  #print("-------------------asdasdas",data['word_bboxes'])
  for image_num in range(data.shape[0]):
    image = cv2.imread(os.path.join(start, data['image_name'][image_num]))
    
    image_name = data['image_name'][image_num].strip('.jpg')
    score_bbox = data['word_bboxes'][image_num].split('),')
    generate_words(image_name, score_bbox, image)


#CRAFT
parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='./CRAFT_pytorch/weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='./CRAFT_pytorch/video_frame/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()


""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args.test_folder)

image_names = []
image_paths = []

#CUSTOMISE START
start = args.test_folder

for num in range(len(image_list)):
  image_names.append(os.path.relpath(image_list[num], start))


result_folder = './CRAFT_pytorch/Results/'

# VIDEO_FRAME(VIDEO_PATH)
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

def video_info_csv():
    VIDEO_FRAME(VIDEO_PATH)

    data=pd.DataFrame(columns=['image_name', 'word_bboxes', 'pred_words', 'align_text'])
    data['image_name'] = image_names

    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(new_test.copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(new_test.copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(new_test.copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(new_test.copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()

    # load data
    for k, image_path in enumerate(natsort.natsorted(image_list)):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text, det_scores = new_test.test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, args, refine_net)
        
        bbox_score={}

        for box_num in range(len(bboxes)):
          key = str (det_scores[box_num])
          item = bboxes[box_num]
          bbox_score[key]=item
        data['word_bboxes'][k]=bbox_score
        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        # mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        # cv2.imwrite(mask_file, score_text)
        
        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)

    print("222",data['word_bboxes'])
    data.to_csv('./CRAFT_pytorch/result/csv_result/data.csv', sep = ',', na_rep='Unknown')
    print("333",data['word_bboxes'])
    print("elapsed time : {}s".format(time.time() - t))

def craft_csv_data():
    video_info_csv()
    image_crop()

# craft_csv_data()
# main()

# video_info_csv()
# image_crop()