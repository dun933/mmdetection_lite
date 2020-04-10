root = "/Users/yanghuiyu/Desktop/baby_fd_val_imgs/*.jpg"
import glob
imgs  = glob.glob(root)
import shutil
import os
os.makedirs("/Users/yanghuiyu/Desktop/baby_fd_val_imgs2")

for img in imgs:
    if not os.path.exists(img.replace("jpg","xml")):
        shutil.move(img,img.replace("baby_fd_val_imgs","baby_fd_val_imgs2"))