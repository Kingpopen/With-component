import imgaug as ia
from imgaug import augmenters as iaa
import imageio
import numpy as np



# 针对分割图片进行数据增强的操作
class AugmentSeg():
    def __init__(self):
        ia.seed(1)
        self.image = ia.quokka(size=(128, 128), extract="square")  # 加载示例图像进来，大小为（128,128,3）
        pass

    def build(self):
        # 创建一个分割的图
        segmap = np.zeros((128, 128), dtype=np.int32)
        segmap[28:71, 35:85] = 1
        segmap[10:25, 30:45] = 2
        segmap[10:25, 70:85] = 3
        segmap[10:110, 5:10] = 4
        segmap[118:123, 10:110] = 5

        # 将图片转换为SegmentationMapOnImage类型
        segmap = ia.SegmentationMapOnImage(segmap, shape=self.image.shape, nb_classes=1 + 5)

        # 定义数据增强方法
        seq = iaa.Sequential([
            iaa.Dropout([0.05, 0.2]),  # drop 5% or 20% of all pixels(丢弃5%-20%的像素点)
            iaa.Sharpen((0.0, 1.0)),  # sharpen the image  锐化图片
            iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees (affects heatmaps) 对图片进行旋转
            iaa.ElasticTransformation(alpha=50, sigma=5)  # apply water effect (affects heatmaps) 弹性变换
        ], random_order=True)

        images_aug = []
        segmaps_aug = []

        # 这里可以通过加入循环的方式，对多张图进行数据增强。
        seq_det = seq.to_deterministic()  # 确定一个数据增强的序列
        images_aug = seq_det.augment_image(self.image)  # 将方法应用在原图像上
        segmaps_aug = seq_det.augment_segmentation_maps([segmap])[0].get_arr_int().astype(np.uint8)
        # 将方法应用在分割标签上，并且转换成np类型

        segmaps_aug = ia.SegmentationMapOnImage(segmaps_aug, shape=self.image.shape, nb_classes=1 + 5)
        # 将分割结果转换为SegmentationMapOnImage类型，方便后面可视化

        # 可视化
        cells = []
        cells.append(self.image)
        cells.append(segmap.draw_on_image(self.image))
        cells.append(images_aug)
        cells.append(segmaps_aug.draw_on_image(images_aug))
        cells.append(segmaps_aug.draw(size=images_aug.shape[:2]))
        grid_image = ia.draw_grid(cells, cols=5)
        imageio.imwrite("./example_segmaps.jpg", grid_image)

if __name__ == '__main__':
    # aug_seg = AugmentSeg()
    # aug_seg.build()
    import imageio
    import numpy as np
    import imgaug as ia
    import imgaug.augmenters as iaa
    from imgaug.augmentables.segmaps import SegmentationMapsOnImage

    ia.seed(1)

    # Load an example image (uint8, 128x128x3).
    image = ia.quokka(size=(128, 128), extract="square")

    # Define an example segmentation map (int32, 128x128).
    # Here, we arbitrarily place some squares on the image.
    # Class 0 is our intended background class.
    '''
    定义一个分割图
    这里，我们随机的在图片上放置一些方块
    0表示为bg
    '''
    segmap = np.zeros((128, 128, 1), dtype=np.int32)
    segmap[28:71, 35:85, 0] = 1
    segmap[10:25, 30:45, 0] = 2
    segmap[10:25, 70:85, 0] = 3
    segmap[10:110, 5:10, 0] = 4
    segmap[118:123, 10:110, 0] = 5
    segmap = SegmentationMapsOnImage(segmap, shape=image.shape)

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # Define our augmentation pipeline.
    seq = iaa.Sequential([
        sometimes(iaa.Dropout([0.05, 0.2])),  # drop 5% or 20% of all pixels（丢掉5%-20%的像素点）
        sometimes(iaa.Sharpen((0.0, 1.0))),  # sharpen the image（锐化图片）
        sometimes(iaa.Affine(rotate=(-45, 45))),# rotate by -45 to 45 degrees (affects segmaps)（对图片进行旋转）
        sometimes(iaa.ElasticTransformation(alpha=50, sigma=5)),# apply water effect (affects segmaps)（）
    ], random_order=True)

    # Augment images and segmaps.（对图片和分割图进行增强）
    images_aug = []
    segmaps_aug = []
    # 总共进行5轮
    for _ in range(10):
        images_aug_i, segmaps_aug_i = seq(image=image, segmentation_maps=segmap)
        images_aug.append(images_aug_i)
        segmaps_aug.append(segmaps_aug_i)

    # We want to generate an image containing the original input image and
    # segmentation maps before/after augmentation. (Both multiple times for
    # multiple augmentations.)
    #
    # The whole image is supposed to have five columns:
    # (1) original image,
    # (2) original image with segmap,
    # (3) augmented image,
    # (4) augmented segmap on augmented image,
    # (5) augmented segmap on its own in.
    #
    # We now generate the cells of these columns.
    #
    # Note that draw_on_image() and draw() both return lists of drawn
    # images. Assuming that the segmentation map array has shape (H,W,C),
    # the list contains C items.
    cells = []
    for image_aug, segmap_aug in zip(images_aug, segmaps_aug):
        cells.append(image)  # column 1
        cells.append(segmap.draw_on_image(image)[0])  # column 2
        cells.append(image_aug)  # column 3
        cells.append(segmap_aug.draw_on_image(image_aug)[0])  # column 4
        cells.append(segmap_aug.draw(size=image_aug.shape[:2])[0])  # column 5

    # Convert cells to a grid image and save.
    grid_image = ia.draw_grid(cells, cols=5)
    imageio.imwrite("example_segmaps.jpg", grid_image)
