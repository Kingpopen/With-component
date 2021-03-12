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
from skimage.measure import find_contours
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from samples.coco.coco import CocoConfig

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
from cococardamage import COCOCarDamage
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
DEFAULT_DATASET_YEAR = "2014"


############################################################
#  Configurations
############################################################

# 配置文件
class CarDamageConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "Cardamage_ali"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    # self.IMAGES_PER_GPU * self.GPU_COUNT
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    # NUM_CLASSES = 1 + 8  # 只需要餐具的种类数即可，不需要材料的种类?
    # NUM_MATERIALS = 1 + 5
    NUM_CLASSES = 1 + 4
    NUM_COMPONENTS = 1 + 6

    BACKBONE = "resnet50"

    # IMAGE_MIN_DIM = IMAGE_MAX_DIM = 128

    # STEPS_PER_EPOCH = 100

    STEPS_PER_EPOCH = 100

    VALIDATION_STEPS = 10

    LEARNING_RATE = 0.001

    def __init__(self):
        super(CarDamageConfig, self).__init__()
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES + self.NUM_COMPONENTS


############################################################
#  Dataset
############################################################

# 数据集导入dataset?

# 输入的数据集跟coco数据集格式类?
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


class CarDamageDataset(utils.Dataset):
    '''
    要使用Mask R-CNN训练自己的数据集，需要写一个自己的MediaDataset类（继承utils.Dataset类），并自己写以下几个方法：
    1. load_media，即读取数据�?
    2. load_mask，即将数据集中的segmentation转化为mask
    下面看到的其他方法都是上述两个方法的附带�?
    '''

    def load_cardamage(self, dataset_dir, subset, year=DEFAULT_DATASET_YEAR, class_ids=None, component_ids=None,
                  class_map=None, return_coco=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """

        coco = COCOCarDamage("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/{}{}".format(dataset_dir, subset, year)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # 添加零件类别
        if not component_ids:
            component_ids = sorted(coco.getCompIds())

        # All images or a subset?  所有图片还是一个子�?
        if component_ids:
            image_ids = []
            for id in component_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add components
        for i in component_ids:
            self.add_component("coco", i, coco.loadComps(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, compIds=component_ids, iscrowd=None)))
        if return_coco:
            return coco


    # 类别的初始化，保证self.component_info中包含所有的�?
    def add_component(self, source, component_id, component_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.component_info:
            if info['source'] == source and info["id"] == component_id:
                # source.class_id combination already available, skip
                return
        # Add the damage
        self.component_info.append({
            "source": source,
            "id": component_id,
            "name": component_name,
        })

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
        if image_info["source"] != "coco":
            return super(CarDamageDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        component_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))

            # 添加零件分支
            component_id = self.map_source_component_id(
                "coco.{}".format(annotation['component_id']))

            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)
                # 添加零件分支
                component_ids.append(component_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            #添加零件分支
            component_ids = np.array(component_ids, dtype=np.int32)
            return mask, class_ids, component_ids
        else:
            # Call super class to return an empty mask
            return super(CarDamageDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return info['path']
        else:
            super(CarDamageDataset, self).image_reference(image_id)
    # The following two functions are from pycocotools with a few changes.

    # RLE是一个微软的压缩格式
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

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


def train(model):
    dataset_train = CarDamageDataset()
    dataset_train.load_cardamage('/home/pengjinbo/kingpopen/Car/dataset1_ali/train/', "train", year='2017')
    dataset_train.prepare()

    num_train = len(dataset_train.image_ids)
    print("num_train:", num_train)
    # print("train dataset ok")

    dataset_val = CarDamageDataset()
    dataset_val.load_cardamage('/home/pengjinbo/kingpopen/Car/dataset1_ali/val/', "val", year='2017')
    dataset_val.prepare()

    num_val = len(dataset_val.image_ids)
    print("num_val:", num_val)

    config.STEPS_PER_EPOCH = num_train // (config.GPU_COUNT * config.IMAGES_PER_GPU)
    config.VALIDATION_STEPS = num_val // (config.GPU_COUNT * config.IMAGES_PER_GPU)
    CarDamageConfig.STEPS_PER_EPOCH = config.STEPS_PER_EPOCH
    CarDamageConfig.VALIDATION_STEPS = config.VALIDATION_STEPS

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    # 在默认的config（MediaConfig类）中，默认将mrcnn_material_loss的权重设�?
    # 也就是说，不训练material的分类，先做好原先的餐具分类、找轮廓的任�?

    # Training - Stage 1
    print("Training network heads")
    #augmentation = imgaug.augmenters.Fliplr(0.5)
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=5,
                layers='heads-no-component')
    print("Stage1 Over")
    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=12,
                layers='4+')
    print("Stage2 Over")
    # 在训练好原有任务后，开始训练material的分类，此时需要建立一个新的config
    # 该config继承了MediaConfig类，并将各个损失的权重做了调?
    # 此时loss会明显上升，毕竟考虑了一个额外的material_loss，让他正常训练就可以?
    class ComponentConfig(CarDamageConfig):
        LOSS_WEIGHTS = {
            "rpn_class_loss": 1.,
            "rpn_bbox_loss": 1.,
            "mrcnn_class_loss": 1.,
            "mrcnn_component_loss": 1.,
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
        # model.load_weights("/home/pengjinbo/kingpopen/Car/With_component/logs/cardamage20210219T2110/mask_rcnn_cardamage_0015.h5", by_name=True)
        return model
    # 实例化MaterialConfig类，根据这个config实例创建新的model
    model = create_load_last(ComponentConfig())
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 2,
                epochs=16,
                layers='component-classifiers')
    print("Stage3 Over")

    # componentconfig = ComponentConfig()
    # model = modellib.MaskRCNN(mode="training", config=componentconfig,
    #                           model_dir=args.logs)
    # model.load_weights(
    #     "/home/pengjinbo/kingpopen/Car/With_component/logs/cardamage20210219T2110/mask_rcnn_cardamage_0015.h5",
    #     by_name=True)
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=20,
    #             layers='component-classifiers')
    # print("Stage3 Over")

    class AlllayerConfig(ComponentConfig):
        '''
        由于显存资源不足，需要在这个时候调低IMAGES_PER_GPU�?
        如果在中途想修改config的内容，可以用这种继承的方法去写�?
        '''

    model = create_load_last(AlllayerConfig())
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=20,
                layers='all')
    print("Stage4 Over")

    # alllayerconfig = AlllayerConfig()
    # model = modellib.MaskRCNN(mode="training", config=alllayerconfig,
    #                           model_dir=args.logs)
    # model.load_weights(
    #     "/home/pengjinbo/kingpopen/Car/With_component/logs/cardamage20210219T2110/mask_rcnn_cardamage_0020.h5",
    #     by_name=True)
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE / 10,
    #             epochs=25,
    #             layers='all',
    #             augmentation=augmentation)
    # print("Stage4 Over")


###########
# Evalution
###########
# mask转化为segmentation（mask是圈内每一个点的坐标，segmentation是圈边缘的坐标）
def mask_to_seg(mask):
    seg = []
    # Mask Polygon
    # Pad to ensure proper polygons for masks that touch image edges.
    padded_mask = np.zeros(
        (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded_mask[1:-1, 1:-1] = mask
    contours = find_contours(padded_mask, 0.5)
    for verts in contours:
        # Subtract the padding and flip (y, x) to (x, y)
        verts = np.fliplr(verts) - 1
        seg.append(verts.tolist())
    return seg


# 将评测的结果保存为json格式
def save_to_json(imageHeight, imageWidth, boxes, masks, class_ids, component_ids,
                 save_name, save_dir):
    class_names = ['__background', 'scratch', 'indentation', 'crack', 'perforation']
    # component_names = ['__background', 'front bumper', 'rear bumper', 'front fender',
    #                    'rear fender', 'door', 'rear taillight', 'headlight',
    #                    'hood', 'luggage cover', 'radiator grille', 'bottom side',
    #                    'rearview mirror', 'license plate']

    component_names = ['__background', "bumper", "fender", "light", "rearview", "windshield", "others"]

    N = boxes.shape[0]
    bbox = []
    segmentation = []
    shapes = []

    for i in range(N):
        mask = masks[:, :, i]
        box = boxes[i]
        box = box.astype(np.float64)
        class_id = class_ids[i]
        component_id = component_ids[i]
        class_name = class_names[class_id]
        component_name = component_names[component_id]

        label = component_name + " " + class_name
        seg = mask_to_seg(mask)
        bbox.append(list(box))
        segmentation.append(seg)

        shape = {
            "label": label,
            "points": seg,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        shapes.append(shape)

    imagePath = save_name.replace('.json', '.jpg')
    data = dict(
        version="4.5.6",
        flags={},
        shapes=shapes,
        imagePath=imagePath,
        imageData=None,
        imageHeight=imageHeight,
        imageWidth=imageWidth,
    )

    json.dump(data, open(os.path.join(save_dir, save_name), 'w', encoding='utf-8'), indent=4)
    # with open(os.path.join(save_dir, save_name), 'w', encoding='utf-8') as f:
    #     json.dump(data, f, indent=2)


def build_coco_results(dataset, image_ids, rois, class_ids, component_ids, scores, scores_components, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            component_id = component_ids[i]

            score = scores[i]
            score_component = scores_components[i]

            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]


            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "component_id": dataset.get_source_component_id(component_id, 'coco'),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "score_component": score_component,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, limit=0, image_ids=None, save_dir=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids
    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    t_prediction = 0
    t_start = time.time()


    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)
        info = dataset.image_info[image_id]

        print("image_id:", image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        #         save
        image_name = info["path"][-19:]
        save_name = image_name.replace('.jpg', '.json').replace('.png', '.json').replace('.bmp', '.json')

        save_to_json(imageHeight=info["height"], imageWidth=info["width"], boxes=r["rois"],
                     masks=r["masks"].astype(np.uint8), class_ids=r["class_ids"], component_ids=r["component_ids"],
                     save_name=save_name, save_dir=save_dir)

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)





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
                        default=10,
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
        config = CarDamageConfig()
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
        config = CarDamageConfig()
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
        model.load_weights(model_path, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
        # model.load_weights(model_path, by_name=True,
        # exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    elif args.model.lower() == "trained149":
        model.load_weights(model_path, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
        # model.load_weights(model_path, by_name=True,
        # exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(model_path, by_name=True)
    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "evaluate":
        # Validation dataset
        model.load_weights(
            '/home/pengjinbo/kingpopen/Car/With_component/logs/cardamage20210228T2129/mask_rcnn_cardamage_0025.h5',
            by_name=True)
        dataset_test = CarDamageDataset()
        coco = dataset_test.load_cardamage('/home/pengjinbo/kingpopen/Car/dataset2/unified_test/', 'unified_test', '2017', return_coco=True)
        dataset_test.prepare()

        print("len of dataset_test:", len(dataset_test.image_ids))

        print("Running COCO evaluation on {} images.".format(args.limit))
        evaluate_coco(model, dataset_test, limit=0, image_ids=None, save_dir="/home/pengjinbo/kingpopen/Car/With_component/for_cardamage/unified_test2_result_json/")
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
