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
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import imageio



DEFAULT_DATASET_YEAR = "2017"
config = Config()
ia.seed(1)

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


def augmentation_demo(image_id, augmentation):
    dataset = CarDamageDataset()
    coco = dataset.load_cardamage('/home/pengjinbo/kingpopen/Car/dataset2/unified_test/', 'unified_test', '2017',
                                       return_coco=True)
    dataset.prepare()

    # Load image and mask
    image = dataset.load_image(image_id)
    # mask, class_ids, material_ids = dataset.load_mask(image_id)
    mask, class_ids, component_ids = dataset.load_mask(image_id)
    original_shape = image.shape
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    mask = utils.resize_mask(mask, scale, padding, crop)

    # Augmentation
    # This requires the imgaug lib (https://github.com/aleju/imgaug)
    if augmentation:
        import imgaug

        # Augmenters that are safe to apply to masks
        # Some, such as Affine, have settings that make them unsafe, so always
        # test your augmentation on masks
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        # Store shapes before augmentation to compare
        image_shape = image.shape
        mask_shape = mask.shape
        print("the shape of mask_shape:", mask_shape)
        # Make augmenters deterministic to apply similarly to images and masks
        # det = augmentation.to_deterministic()
        # image = det.augment_image(image)
        # # Change mask to np.uint8 because imgaug doesn't support np.bool
        # mask = det.augment_image(mask.astype(np.uint8),
        #                          hooks=imgaug.HooksImages(activator=hook))

        segmap = SegmentationMapsOnImage(mask[:, :, 1], shape=image.shape)
        # Augment images and segmaps.
        images_aug = []
        segmaps_aug = []
        for _ in range(5):
            images_aug_i, segmaps_aug_i = seq(image=image, segmentation_maps=segmap)
            images_aug.append(images_aug_i)
            segmaps_aug.append(segmaps_aug_i)
        cells = []
        for image_aug, segmap_aug in zip(images_aug, segmaps_aug):
            cells.append(image)  # column 1
            cells.append(segmap.draw_on_image(image)[0])  # column 2
            cells.append(image_aug)  # column 3
            cells.append(segmap_aug.draw_on_image(image_aug)[0])  # column 4
            cells.append(segmap_aug.draw(size=image_aug.shape[:2])[0])  # column 5

        # Convert cells to a grid image and save.
        grid_image = ia.draw_grid(cells, cols=5)
        imageio.imwrite("./example_segmaps.jpg", grid_image)


        print("the shape of mask is:", mask.shape)
        print("mask[:,:,0]:", mask[:, :, 0])
        print("mask[:,:,1]:", mask[:, :, 1])
        print("the type of mask:", type(mask))
        print("class of mask[:,:,0]:", np.unique(mask[:, :, 0]))
        print("class of mask[:,:,1]", np.unique(mask[:, :, 1]))


        # Verify that shapes didn't change
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
        # Change mask back to bool
        # mask = mask.astype(np.bool)




if __name__ == '__main__':
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    image_id = 1

    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # drop 5% or 20% of all pixels
        iaa.Flipud(0.5),  # sharpen the image
        sometimes(iaa.Affine(                          
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-45, 45),  
            shear=(-16, 16),    
            order=[0, 1],   
            cval=(0, 255),  
            mode=ia.ALL   
        )),
    ], random_order=True)

    augmentation_demo(image_id, augmentation=seq)


