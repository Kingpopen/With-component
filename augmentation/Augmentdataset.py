import traceback

from mrcnn.config import Config
from mrcnn import model as modellib, utils
from cococardamage import COCOCarDamage
import os
import sys
import time
import numpy as np
import json
from pycocotools import mask as maskUtils
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import imageio
import skimage
import math


DEFAULT_DATASET_YEAR = "2017"
config = Config()
ia.seed(2)

# 160分类的id与名称对应关系
label_160_value_to_name = { 1:"front bumper", 2:"rear bumper",
            3:"front bumper grille", 4:"front windshield",
            5:"rear windshield", 6:"front tire",
            7:"rear tire", 8:"front side glass",
            9:"rear side glass", 10:"front fender",
            11:"rear fender", 12:"front mudguard",
            13:"rear mudguard", 14:"turn signal",
            15:"front door", 16:"rear door",
            17:"rear outer taillight", 18:"rear inner taillight",
            19:"headlight", 20:"fog light",
            21:"hood", 22:"luggage cover",
            23:"roof", 24:"steel ring",
            25:"radiator grille", 26:"a pillar",
            27:"b pillar", 28:"c pillar",
            29:"d pillar", 30:"bottom side",
            31:"rearview mirror", 32:"license plate"}

label_52_value_to_name = {1: "front bumper", 2: "rear bumper", 3: "front fender",
                        4: "rear fender", 5: "door", 6: "rear taillight", 7: "headlight",
                        8: "hood", 9: "luggage cover", 10: "radiator grille", 11: "bottom side",
                        12: "rearview mirror", 13: "license plate"
}

label_24_value_to_name = {1: "bumper", 2:"fender", 3:"light", 4:"rearview",
                          5:"windshield", 6:"others"
}

class CarDamageDataset(utils.Dataset):

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

        if not component_ids:
            component_ids = sorted(coco.getCompIds())

        # All images or a subset?
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



class AugmentationDataset():
    def __init__(self, dataset, augmentation, image_ids):
        self.image_ids = image_ids
        self.dataset = dataset
        self.augmentation = augmentation
        self.savedir = "/home/pengjinbo/kingpopen/Car/augmentation/"
        
        print("img ids :", self.dataset.image_ids)
        
        self.mhd_e = 30




    def build(self):
        for image_id in image_ids:
            image = self.dataset.load_image(image_id)
            mask, class_ids, component_ids = self.dataset.load_mask(image_id)
            filename = self.dataset.source_image_link(image_id)[-19:]
            # print("filename:", filename)
            print("image_id:", image_id)
            # Store shapes before augmentation to compare
            image_shape = image.shape
            mask_shape = mask.shape

            # Make augmenters deterministic to apply similarly to images and masks
            image, mask = self.img2aug_img(image, mask)
            Height, Width, _ = image.shape

            assert image.shape == image_shape, "Augmentation shouldn't change image size"
            assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"

            try:
                self.mask2json(image, mask, class_ids, component_ids, filename, Height, Width)
            except Exception:
                print("filename:", filename)
                traceback.print_exc()


    # 进行数据增强
    def img2aug_img(self, image, mask):
        det = self.augmentation.to_deterministic()
        image = det.augment_image(image)
        # Change mask to np.uint8 because imgaug doesn't support np.bool
        mask = det.augment_image(mask.astype(np.uint8))
        #print("the shape of mask:", mask.shape)
        return image, mask

    # 优化
    def optimize(self, shapes):

        data = self.__xl(shapes)
        # 再进行曼哈顿距离处理
        data = self.__mhd(data)
        return data

    # 将获取的mask转化为json
    def mask2json(self, image, mask, class_ids, component_ids, filename, imageHeight, imageWidth):
        num = mask.shape[2]
        #print("class_ids:", class_ids)
        #print("component_ids:", component_ids)

        shapes = []
        for index in range(num):
            instance = mask[:, :, index]
            # origin label file don‘t consider the background�?for example scratch is labeled as 0
            category = class_ids[index] - 1
            component = component_ids[index]

            contours, hierarchy = cv2.findContours(instance, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # print("contours:", contours)
            #print("the shape of contours:", np.array(contours).shape)
            contour = contours[0]
            points = []
            contour = contour.reshape(-1, 2)
            for list in contour:
                point = []
                point.append(float(list[0]))
                point.append(float(list[1]))
                points.append(point)

            shape = {
                "label": label_24_value_to_name[component],
                "points": points,
                "group_id": int(category),
                "shape_type": "polygon",
                "flags": {}
            }
            shapes.append(shape)

        shapes = self.optimize(shapes)
        self.__save(image, filename, shapes, imageHeight, imageWidth)


    # 保存
    def __save(self, image, filename, shapes, imageHeight, imageWidth, imageData=None, otherData=None, flags=None,):
        if otherData is None:
            otherData = {}
        if flags is None:
            flags = {}
        data = dict(
            version="4.5.6",
            flags=flags,
            shapes=shapes,
            imagePath="aug_" + filename,
            imageData=imageData,
            imageHeight=imageHeight,
            imageWidth=imageWidth,
        )
        for key, value in otherData.items():
            assert key not in data
            data[key] = value

        filename = filename[:-4]
        json_save_name = os.path.join(self.savedir, "aug_" + filename + ".json")
        image_save_name = os.path.join(self.savedir, "aug_" + filename + ".jpg")

        with open(json_save_name, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        skimage.io.imsave(image_save_name, image)

    # 斜率
    def __xl(self, shapes):

        len_shape = len(shapes)
        dirty = []
        per = 0
        for i in range(len_shape):
            points = shapes[i]["points"]
            len_point = len(points)
            if len_point <= 15:
                dirty.append(i)
                continue
            dx, dy = points[0][0] - points[1][0], points[0][1] - points[1][1]
            tmp = []
            for j in range(1, len_point):
                dx1, dy1 = points[j][0] - points[(j + 1) % len_point][0], points[j][1] - points[(j + 1) % len_point][1]
                if dx1 * dy != dy1 * dx:
                    tmp.append(points[j])
                dx, dy = dx1, dy1
            if len(tmp) <= 15:
                dirty.append(i)
                continue
            # print(len(tmp), len_point)
            per += (len(tmp) / len_point)
            shapes[i]["points"] = tmp

        for i in range(len(dirty)):
            del shapes[dirty[i] - i]
        # if (len_shape - len(dirty)) != 0:
        #     per /= (len_shape - len(dirty))
        #     #print("斜率处理后保留：", per * 100, "%的点")
        # else:
        #     print("全都删完了~")
        return shapes

    # 曼哈顿距�?
    def __mhd(self, shapes):
        len_shape = len(shapes)
        dirty = []
        per = 0
        for i in range(len_shape):
            points = shapes[i]["points"]
            len_point = len(points)
            # print("len_point:", len_point)

            if len_point <= 5:
                dirty.append(i)
                continue

            tmp = []
            index = 0
            for j in range(0, len_point):
                dx1, dy1 = points[index][0] - points[(j + 1) % len_point][0], points[index][1] - \
                           points[(j + 1) % len_point][1]
                if math.fabs(dx1) + math.fabs(dy1) >= self.mhd_e:
                    tmp.append(points[(j + 1) % len_point])
                    index = j + 1

            if len(tmp) <= 5:
                dirty.append(i)
                continue
            per += (len(tmp) / len_point)
            shapes[i]["points"] = tmp

        for i in range(len(dirty)):
            del shapes[dirty[i] - i]

        # if (len_shape - len(dirty)) !=0:
        #     per /= (len_shape - len(dirty))
        #     print("曼哈顿处理后再保留：", per * 100, "%的点")
        # else:
        #     print("全都删完了~")

        return shapes

if __name__ == '__main__':
    dataset_path = '/home/pengjinbo/kingpopen/Car/new_dataset/ali_dataset_multi/train'
    datset_path = dataset_path
    dataset = CarDamageDataset()
    dataset.load_cardamage(dataset_path, "train", year='2017')
    dataset.prepare()
    image_ids = dataset.image_ids

    image_ids = image_ids

    sometimes = lambda aug: iaa.Sometimes(0.8, aug)
    seq = iaa.Sequential([
                    iaa.Fliplr(0.5),  # 左右翻转
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        rotate=(-45, 45),
                        shear=(-16, 16),
                        #order=[0, 1],
                        #cval=(0, 255),
                        #mode=ia.ALL
                    ),
                    # sometimes(iaa.ElasticTransformation(alpha=50, sigma=5))
                ], random_order=True)
    augdataset = AugmentationDataset(dataset, seq, image_ids)
    augdataset.build()