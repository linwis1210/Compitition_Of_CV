# -----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
# -----------------------------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image
from utils.utils import get_classes
from yolo import YOLO

if __name__ == "__main__":
    # 加载模型
    yolo = YOLO()
    # ----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    # ----------------------------------------------------------------------------------------------------------#
    mode = "dir_predict"
    # -------------------------------------------------------------------------#
    #   crop                指定了是否在单张图片预测后对目标进行截取
    #   count               指定了是否进行目标的计数
    #   crop、count仅在mode='predict'时有效
    # -------------------------------------------------------------------------#
    crop = False
    count = False
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    # -------------------------------------------------------------------------#
    dir_origin_path = "img/JPEGImages/"
    dir_save_path = "text_out/"
    # -------------------------------------------------------------------------#
    #   classes_path            类别路径
    # -------------------------------------------------------------------------#
    classes_path = 'model_data/cls_classes.txt'
    class_names, num_classes = get_classes(classes_path)

    if mode == "dir_predict":
        import os

        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                # r_image = yolo.detect_image(image)
                if not os.path.exists(os.path.join(dir_save_path, "detection-results")):
                    os.makedirs(os.path.join(dir_save_path, "detection-results"))
                yolo.get_map_txt(image_id=img_name[:-4], image=image, class_names=class_names, map_out_path=dir_save_path)

                # r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)
    else:
        raise AssertionError(
            "Please specify the correct mode: 'dir_predict'.")
