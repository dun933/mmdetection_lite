from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result
import mmcv
import glob
import os
from tqdm import  tqdm
import shutil
import numpy as np
import cv2
import os
import sys
import shutil
if os.path.exists("./result"):
    shutil.rmtree("./result")
os.mkdir("./result")

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0,current_dir)

# config_file = '../configs/thundernet/thundernet_voc_shufflenetv2.py'
# config_file = '../configs/thundernet/thundernet_voc_mobilenetv2.py'
config_file = '../configs/thundernet/thundernet_coco_mobilenetv2.py'
# config_file = '../configs/mask_rcnn_lite/mask_rcnn_lite_mobilenetv2.py'
# config_file = '../configs/cascsde_rcnn_light/cascade_rcnn_shufflenetv2_1x_fpn_1x_voc.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# checkpoint_file = '../work_dirs/cascade_rcnn_shufflenetv2_1x_fpn_1x_voc/latest.pth'
# checkpoint_file = '../work_dirs/thundernet_voc_shufflenetv2/latest.pth'
checkpoint_file = '../work_dirs/thundernet_coco_mobilenetv2/latest.pth'
# checkpoint_file = '../work_dirs/mask_rcnn_lite_mobilenetv2/latest.pth'

#%%

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

vocroot = "/mnt/data1/yanghuiyu/datas/voc0712/VOC/VOCdevkit/VOC2007/JPEGImages/"
voc_test = "/mnt/data1/yanghuiyu/datas/voc0712/VOC/VOCdevkit/VOC2007/ImageSets/Main/test.txt"
txts = open(voc_test).readlines()
voc_imgs = [os.path.join(vocroot,x.strip() + ".jpg")  for x in txts]


# test a single image
# imgs  = glob.glob("/mnt/data2/yanghuiyu/taobao_live/train_data/detect/coco/images/val/*.jpg")
imgs  = glob.glob("/mnt/data1/yanghuiyu/datas/coco/coco/images/val2017/*.jpg")


count = 0
for img in tqdm(imgs[::200],ncols=50):

    result = inference_detector(model, img)
    # print(result)
    show_result(img, result, model.CLASSES ,score_thr=0.35 , out_file = "./result/count_{}.jpg".format(count))
    count += 1
