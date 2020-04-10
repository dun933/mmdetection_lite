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

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0,current_dir)

config_file = './mmdetection/configs/cascade_rcnn_r50_fpn_1x.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = './mmdetection/work_dirs/cascade_rcnn_r50_fpn_1x/latest.pth'

#%%

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

#%%

CLASSES = ['短袖上衣', '长袖上衣', '短袖衬衫', '长袖衬衫', '背心上衣', '吊带上衣', '无袖上衣', '短外套', '短马甲', '长袖连衣裙', '短袖连衣裙', '无袖连衣裙', '长马甲', '长外套', '连体衣', '古风', '古装', '短裙', '中等半身裙', '长半身裙', '短裤', '中裤', '长裤', '背带裤']




# test a single image
# imgs  = glob.glob("/mnt/data2/yanghuiyu/taobao_live/train_data/detect/coco/images/val/*.jpg")
#

#
# for img in tqdm(imgs[::100],ncols=50):
#
#     result = inference_detector(model, img)
#     bboxes = np.vstack(result)
#     # print(bboxes.tolist())
#     labels = [
#         np.full(bbox.shape[0], i, dtype=np.int32)
#         for i, bbox in enumerate(result)
#     ]
#     im = cv2.imread(img)
#     labels = np.concatenate(labels)
#     for box,label  in zip(bboxes,labels):
#         x1,y1,x2,y2 ,score = box
#         x1, y1, x2, y2 = list(map(int,[x1,y1,x2,y2 ]))
#         if score < 0.3 :
#             continue
#         # crop_im = im[y1:y2,x1:x2,:]
#         # cv2.imwrite("./{}_{}.jpg".format(label,str(score)),crop_im)
#     break
#     # show_result(img, result, CLASSES ,score_thr=0.25 , out_file = "./result/{}".format(os.path.basename(img)))
#

def detect(img_path,thresh = 0.25 ):
    result = inference_detector(model, img_path)
    bboxes = np.vstack(result)

    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(result)
    ]

    labels = np.concatenate(labels)
    res = []
    for box,label  in zip(bboxes,labels):
        x1,y1,x2,y2 ,score = box
        if score < thresh:
            continue
        res.append([x1,y1,x2,y2 ,score,label])
    return  res