"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import sys
import time
import numpy as np
# import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)

from imgaug import augmenters as iaa #引入数据增强的包

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
# config.gpu_options.visible_device_list = "1"
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = tf.Session(config=config)
KTF.set_session(session)

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil
import json
from pycocotools import mask as maskUtils
import warnings
warnings.filterwarnings("ignore")

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

Midea_Model_Path = os.path.join(ROOT_DIR, "mask_rcnn_media_0244.h5")


# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

#配置文件
class MediaConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "Media"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    # self.IMAGES_PER_GPU * self.GPU_COUNT
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 4

    # Number of classes (including background)
    # NUM_CLASSES = 1 + 8  # 只需要餐具的种类数即可，不需要材料的种类数
    # NUM_MATERIALS = 1 + 5
    NUM_CLASSES = 1 + 4
    NUM_MATERIALS = 1 + 4

    BACKBONE = "resnet101"

    # IMAGE_MIN_DIM = IMAGE_MAX_DIM = 128

    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 10

    LEARNING_RATE = 0.001

############################################################
#  Dataset
############################################################

#数据集导入dataset类

# 输入的数据集跟coco数据集格式类似
'''
"images":...
"annotations":
            [{"id"},
            {"image_id"},
            {"category_id"},
            {"material_id"},
            {"segmentation"},
            {"bbox"},
            {"iscrowd"},]
...    
'''

class MediaDataset(utils.Dataset):
    '''
    要使用Mask R-CNN训练自己的数据集，需要写一个自己的MediaDataset类（继承utils.Dataset类），并自己写以下几个方法：
    1. load_media，即读取数据集
    2. load_mask，即将数据集中的segmentation转化为mask
    下面看到的其他方法都是上述两个方法的附带品
    '''
    def add_material(self, source, material_id, material_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.material_info:
            if info['source'] == source and info["id"] == material_id:
                # source.class_id combination already available, skip
                return

        # Add the class（material_info中没有该类别，就添加进去...）
        self.material_info.append({
            "source": source,
            "id": material_id,
            "name": material_name,
        })


    # 导入类似COCO的数据集格式（dataset_dir是数据集的路径， subset是数据集类型：train or val）
    def load_media(self, dataset_dir, subset):
        # 如果要增加材料或餐具的类，在这里加
        self.add_class("media", 1, "bowl")
        self.add_class("media", 2, "cup")
        self.add_class("media", 3, "plate")
        self.add_class("media", 4, "red-wine")

        # self.add_class("media", 5, "cup body")
        # self.add_class("media", 6, "bottom of cup")
        # self.add_class("media", 7, "frying pan")
        # self.add_class("media", 8, "wok")
        
        # self.add_class("media", 1, "Rear&cup&holder")
        # self.add_class("media", 2, "Front&right&cup&holder")
        # self.add_class("media", 3, "Front&left&cup&holder")
        self.add_material("media", 1, "pottery and porcelain")
        self.add_material("media", 2, "glass")
        self.add_material("media", 3, "stainless steel")
        self.add_material("media", 4, "plastic")

        # self.add_material("media", 5, "cutter")

        # self.add_material("media", 2, "cutter")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        # js = json.load(open(os.path.join(dataset_dir, "data.json")))
        with open(os.path.join(dataset_dir, "data.json")) as f:
            js = json.load(f)
        images = js['images']
        anns = js['annotations']
        for i in range(len(images)):
            image = images[i]
            self.add_image("media",
                           image_id=image['id'],
                           path=os.path.join(dataset_dir, image['file_name']),
                           #  class_id=a['category_id'],
                           width=image['width'],
                           height=image['height'],
                           annotations=anns[i])

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]

        instance_masks = []
        class_ids = []
        material_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "media.{}".format(annotation['category_id']))
            material_id = self.map_source_material_id(
                "media.{}".format(annotation['material_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                instance_masks.append(m)
                class_ids.append(class_id)
                material_ids.append(material_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            material_ids = np.array(material_ids, dtype=np.int32)
            return mask, class_ids, material_ids
        else:
            # Call super class to return an empty mask
            return super(MediaDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "media":
            return info['path']
        else:
            super(MediaDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle


def train(model):
    dataset_train = MediaDataset()
    dataset_train.load_media(args.dataset, "train")
    dataset_train.prepare()

    dataset_val = MediaDataset()
    dataset_val.load_media(args.dataset, "val")
    dataset_val.prepare()
 
    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    # 在默认的config（MediaConfig类）中，默认将mrcnn_material_loss的权重设为0
    # 也就是说，不训练material的分类，先做好原先的餐具分类、找轮廓的任务
    print("Training network heads")
    # augmentation = imgaug.augmenters.Fliplr(0.5)
    augmentation = iaa.Fliplr(0.5)

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=60,
                layers='heads-no-material')
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=150,
                layers='4+',
                augmentation=augmentation)

    # 在训练好原有任务后，开始训练material的分类，此时需要建立一个新的config
    # 该config继承了MediaConfig类，并将各个损失的权重做了调整
    # 此时loss会明显上升，毕竟考虑了一个额外的material_loss，让他正常训练就可以了
    class MaterialConfig(MediaConfig):
        LOSS_WEIGHTS = {
            "rpn_class_loss": 1.,
            "rpn_bbox_loss": 1.,
            "mrcnn_class_loss": 1.,
            "mrcnn_material_loss": 1.,
            "mrcnn_bbox_loss": 1.,
            "mrcnn_mask_loss": 1.
        }

    def create_load_last(config):
        '''
        根据新的config，创建新的model
        :param config:
        :return:
        '''
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
        model_path = model.find_last()
        model.load_weights(model_path, by_name=True)
        return model

    # 实例化MaterialConfig类，根据这个config实例创建新的model
    model = create_load_last(MaterialConfig())
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=200,
                layers='material-classifiers')

    class AlllayerConfig(MaterialConfig):
        '''
        由于显存资源不足，需要在这个时候调低IMAGES_PER_GPU。
        如果在中途想修改config的内容，可以用这种继承的方法去写。
        '''


    model = create_load_last(AlllayerConfig())
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=250,
                layers='all',
                augmentation=augmentation)

############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--download', required=False,
                        default=False,
                        metavar="<True|False>",
                        help='Automatically download and unzip MS-COCO files (default=False)',
                        type=bool)
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    print("Auto Download: ", args.download)

    # Configurations
    if args.command == "train":
        config = MediaConfig()
    else:
        class InferenceConfig(CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    # config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "trained149":
        # Start from ImageNet trained weights
        model_path = Midea_Model_Path
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    if args.model.lower() == "coco":
        model.load_weights(model_path, by_name = True,
        exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
        #model.load_weights(model_path, by_name=True,
                           #exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    elif args.model.lower() == "trained149":
        model.load_weights(model_path, by_name=True,
        exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
        #model.load_weights(model_path, by_name=True,
                           #exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(model_path, by_name=True)
    # Train or evaluate
    if args.command == "train":
        train(model)
