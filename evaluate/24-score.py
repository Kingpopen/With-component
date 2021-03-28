import shutil
from PIL import Image
import argparse
import base64
import json
import os
import numpy as np
import cv2
from labelme import utils
import traceback


new_damage_value_to_name = {
    1: "scratch",
    2: "indentation",
    3: "crack and perforation",
    4: "gap",
    5: "light scratch",
    6: "light perforation",
    7: "rearview scratch",
    8: "rearview perforation",
    9: "rearview crack",
    10: "windshield perforation"
}

aliyun_name_to_value = {
    "_background_": 0,
    "scratch": 1,
    "indentation": 2,
    "crack and perforation": 3,
    "gap": 4,
    "light scratch": 5,
    "light perforation": 6,
    "rearview scratch": 7,
    "rearview perforation": 8,
    "rearview crack": 9,
    "windshield perforation": 10
}

label_name_to_value = {"_background_":0,
            "front bumper":1, "rear bumper":2,
            "front bumper grille":3, "front windshield":4,
            "rear windshield":5, "front tire":6,
            "rear tire":7, "front side glass":8,
            "rear side glass":9, "front fender":10,
            "rear fender":11, "front mudguard":12,
            "rear mudguard":13, "turn signal":14,
            "front door":15, "rear door":16,
            "rear outer taillight":17, "rear inner taillight":18,
            "headlight":19, "fog light":20,
            "hood":21, "luggage cover":22,
            "roof":23, "steel ring":24,
            "radiator grille":25, "a pillar":26,
            "b pillar":27, "c pillar":28,
            "d pillar":29, "bottom side":30,
            "rearview mirror":31, "license plate":32}

baidu_name_to_value = {"__background": 0,
                       "front bumper": 1, "rear bumper": 2,
                       "front fender": 3,
                       "rear fender": 4,
                       "door": 5,
                       "rear taillight": 6,
                       "headlight": 7,
                       "hood": 8,
                       "luggage cover": 9,
                       "radiator grille": 10,
                       "bottom side": 11,
                       "rearview mirror": 12,
                       "license plate": 13
                       }

damage_name_to_value_24 = {
    "__background scratch": 0,
    "__background indentation": 0,
    "__background crack": 0,
    "__background perforation": 0,

    "bumper scratch": 1,
    "bumper indentation": 2,
    "bumper crack": 4,
    "bumper perforation": 3,

    "fender scratch": 1,
    "fender indentation": 2,
    "fender crack": 4,
    "fender perforation": 3,

    "light scratch": 5,
    "light indentation": 2,
    "light crack": 3,
    "light perforation": 6,

    "rearview scratch": 7,
    "rearview indentation": 2,
    "rearview crack": 9,
    "rearview perforation": 8,

    "windshield scratch": 1,
    "windshield indentation": 2,
    "windshield crack": 3,
    "windshield perforation": 10,

    "others scratch": 1,
    "others indentation": 2,
    "others crack": 3,
    "others perforation": 3,
}

seed = 10

# 标注的json文件路径（原始160分类）
jsonpath = "json_label"
# 存放转换后标注的json文件的路径
label_json_path = "json_label_10"
# 预测json文件的路径
predict_json_path = "infer_24_new"

# 存放标注mask的路径   ps:这两个存放mask的文件夹要先建好
label_mask_path = "label_mask"
# 存放预测mask的路径
predict_mask_path = "predict_mask"

# 存放结果1
result1 = "result1.txt"
# 存放结果2
result2 = "result2.txt"


# 将label_json转换为阿里云10分类json
def tranform_160_to_10():
    files = os.listdir(jsonpath)
    despath = label_json_path
    wrongpath = "wrong"

    for filename in files:
        data = json.load(open(os.path.join(jsonpath, filename), encoding="UTF-8"))
        shapes = data["shapes"]
        temp = shapes[:]
        # 筛选损伤类别
        shape_num = 0
        for shape in temp:
            new_damage_id = 0
            try:
                group_id = shape["group_id"]
                label = shape["label"]
                label_id = label_name_to_value[label]

                # 判断160种损伤类型
                # 除去的类
                if label_id in [12, 13, 14, 20, 23, 24, 26, 27, 28, 29]:
                    del shapes[shape_num]
                    continue

                if group_id == 0:
                    # 除去的类
                    if label_id in [3, 4, 5, 6, 7, 8, 9, 25]:
                        del shapes[shape_num]
                        continue
                    # 5.大灯破损
                    if label_id in [17, 18, 19]:
                        new_damage_id = 5
                    # 7.后视镜刮擦
                    if label_id == 31:
                        new_damage_id = 7
                    # 1.刮擦
                    if new_damage_id == 0:
                        new_damage_id = 1

                # 2.凹陷
                if group_id == 1:
                    # 除去的类
                    if label_id in [3, 4, 5, 6, 7, 8, 9, 17, 18, 19, 31]:
                        del shapes[shape_num]
                        continue
                    new_damage_id = 2

                if group_id == 2:
                    # 除去的类
                    if label_id in [6, 7]:
                        del shapes[shape_num]
                        continue
                    # 4.缝隙
                    if label_id in [1, 2, 10, 11]:
                        new_damage_id = 4
                    # 10.挡风玻璃破损
                    if label_id in [4, 5]:
                        new_damage_id = 10
                    # 6.大灯破损
                    if label_id in [17, 18, 19]:
                        new_damage_id = 6
                    # 9.后视镜脱落
                    if label_id == 31:
                        new_damage_id = 9
                    # 3.破洞
                    if new_damage_id == 0:
                        new_damage_id = 3

                if group_id == 3:
                    # 10.挡风玻璃破损
                    if label_id in [4, 5]:
                        new_damage_id = 10
                    # 6.大灯破损
                    if label_id in [17, 18, 19]:
                        new_damage_id = 6
                    # 8.后视镜破损
                    if label_id == 31:
                        new_damage_id = 8
                    # 3.破洞
                    if new_damage_id == 0:
                        new_damage_id = 3

                if group_id == 4:
                    # 10.挡风玻璃破损
                    if label_id in [4, 5]:
                        new_damage_id = 10
                    # 6.大灯破损
                    if label_id in [17, 18, 19]:
                        new_damage_id = 6
                    # 9.后视镜破损
                    if label_id == 31:
                        new_damage_id = 9
                    # 除去的类
                    if new_damage_id == 0:
                        del shapes[shape_num]
                        continue

                # 修改shape信息
                shape_num += 1
                shape["label"] = new_damage_value_to_name[new_damage_id]
                shape["group_id"] = None

            except ValueError or KeyError:
                print("filename:", filename)
                shutil.copyfile(os.path.join(jsonpath ,filename), os.path.join(wrongpath, filename))
                traceback.print_exc()

        # 对结果进行保存
        data["shapes"] = shapes
        filepath = os.path.join(despath, filename)
        with open(filepath, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# 制作mask
def make_mask(json_path, mask_path, name_to_value):
    '''
    json_path 是原来的10分类的阿里云格式json
    mask_path 是生成的mask保存路径（语义分割类型mask）
    '''
    files = os.listdir(json_path)

    # 制作预测的mask标签
    for jsonfile in files:
        filename = jsonfile[:-5]
        json_file = os.path.join(json_path, jsonfile)
        try:
            data = json.load(open(json_file))
        except ValueError:
            print("open failed:", filename)
            continue

        image_data = json.load(open(os.path.join(label_json_path, jsonfile)))
        imageData = image_data.get("imageData")
        if not imageData:
            print('none')
        else:
            img = utils.img_b64_to_arr(imageData)
        try:
            lbl = utils.shapes_to_label(img.shape, data["shapes"], name_to_value)
        except AssertionError:
            print("get mask failed:", filename)
        else:
            w, h = lbl.shape
            # print(lbl)
            # class_index = np.unique(lbl)
            # print(class_index)
            cv2.imwrite(os.path.join(mask_path, filename + ".png"), lbl.reshape(w, h, -1))
            # print(filename + ".mask is completed")


def count_aliyun(name_to_value):
    # 根据是否识别计算

    # 统计各损伤数目
    right_damage = [0] * 10
    label_damage = [0] * 10
    predict_damage = [0] * 10

    files = os.listdir(predict_json_path)

    for filename in files:
        try:
            predict_data = json.load(open(os.path.join(predict_json_path, filename), encoding="gbk"))
        except:
            print('open json failed:' + filename)
        else:
            label_data = json.load(open(os.path.join(label_json_path, filename), encoding="gbk"))

            predict_shapes = predict_data["shapes"]
            label_shapes = label_data["shapes"]

            # 记录是否有损伤
            label_flag = [0] * 10
            predict_flag = [0] * 10

            # 记录损伤
            for shape in predict_shapes:
                label = shape["label"]
                label_id = name_to_value[label]
                if label_id != 0:
                    predict_flag[label_id-1] = 1
            for shape in label_shapes:
                label = shape["label"]
                label_id = aliyun_name_to_value[label]
                label_flag[label_id-1] = 1

            # 损伤求和
            for i in range(10):
                predict_damage[i] += predict_flag[i]
                label_damage[i] += label_flag[i]
                if predict_flag[i] == label_flag[i]:
                    right_damage[i] += label_flag[i]

            print('count1 over:' + filename)

    # 计算准确率和召回率
    acc = [0] * 10
    rec = [0] * 10
    all_num = 0
    acc_mean = 0
    rec_mean = 0
    with open(result1, 'w') as f:
        f.write('label损伤图片数：' + ' ')
        for i in range(10):
            f.write(str(label_damage[i]) + ' ')
        f.write('\n')
        f.write('predict损伤图片数：' + ' ')
        for i in range(10):
            f.write(str(predict_damage[i]) + ' ')
        f.write('\n')
        f.write('正确损伤图片数：' + ' ')
        for i in range(10):
            f.write(str(right_damage[i]) + ' ')
        f.write('\n')
        f.write('准确率：' + ' ')
        for i in range(10):
            if predict_damage[i] == 0:
                acc[i] = 0
            else:
                acc[i] = right_damage[i] / predict_damage[i]
            all_num += label_damage[i]
            acc_mean += acc[i] * label_damage[i]
            f.write(str('%.4f' % acc[i]) + ' ')
        f.write('\n')
        f.write('召回率：' + ' ')
        for i in range(10):
            rec[i] = right_damage[i] / label_damage[i]
            rec_mean += rec[i] * label_damage[i]
            f.write(str('%.4f' % rec[i]) + ' ')
        f.write('\n')
        f.write('平均准确率和召回率：' + ' ')
        f.write(str('%.4f' % (acc_mean/all_num)) + ' ')
        f.write(str('%.4f' % (rec_mean/all_num)))

    print(acc)
    print(rec)

    # 根据像素点计算
    files = os.listdir(predict_mask_path)

    f1_count = []
    precision_count = []
    recall_count = []

    TP_num = []
    precision_num = []
    label_num = []

    mix_array = [[0 for i in range(11)] for j in range(11)]

    for i in range(10):
        f1_count.append(0)
        precision_count.append(0)
        recall_count.append(0)

        TP_num.append(0)
        precision_num.append(0)
        label_num.append(0)

    for mask in files:
        filename = mask[:-4]
        label_path = os.path.join(label_mask_path, mask)
        predict_path = os.path.join(predict_mask_path, mask)

        gray1 = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        gray2 = cv2.imread(predict_path, cv2.IMREAD_GRAYSCALE)

        w, h = gray1.shape
        right_array = np.arange(w*h).reshape(w, h)
        for i in range(w):
            for j in range(h):
                row = gray1[i][j]
                col = gray2[i][j]
                mix_array[row][col] += 1

                if gray1[i][j] != gray2[i][j]:
                    right_array[i][j] = 0
                else:
                    right_array[i][j] = gray1[i][j]

        for i in range(1, 11):
            TP = np.sum(right_array == i)
            label = np.sum(gray1 == i)
            Prediction = np.sum(gray2 == i)

            TP_num[i - 1] += TP
            label_num[i - 1] += label
            precision_num[i - 1] += Prediction

        print("count2 over:" + filename)

    for i in range(10):
        if precision_num[i] != 0:
            precision_count[i] = TP_num[i] / precision_num[i]
        else:
            precision_count[i] = -1
        if label_num[i] != 0:
            recall_count[i] = TP_num[i] / label_num[i]
            if TP_num[i] == 0:
                f1_count[i] = 0
            else:
                f1_count[i] = 2 * (precision_count[i] * recall_count[i]) / (precision_count[i] + recall_count[i])
        else:
            recall_count[i] = -1
            f1_count[i] = -1

    all_n = 0
    acc_mean = 0
    rec_mean = 0
    f1_mean = 0

    print(precision_count)
    print(recall_count)
    print(f1_count)

    with open(result2, 'w') as f:
        f.write('准确率：' + ' ')
        for i in range(10):
            all_n += label_damage[i]
            acc_mean += precision_count[i] * label_damage[i]
            f.write(str('%.4f' % precision_count[i]) + ' ')
            # print('%.4f' % precision_count[i])
        f.write('\n')
        f.write('召回率：' + ' ')
        for i in range(10):
            rec_mean += recall_count[i] * label_damage[i]
            f.write(str('%.4f' % recall_count[i]) + ' ')
            # print('%.4f' % recall_count[i])
        f.write('\n')
        f.write('f1：' + ' ')
        for i in range(10):
            f1_mean += f1_count[i] * label_damage[i]
            f.write(str('%.4f' % f1_count[i]) + ' ')
            # print('%.4f' % f1_count[i])
        f.write('\n')
        f.write('平均值：' + ' ')
        f.write(str('%.4f' % (acc_mean/all_n)) + ' ' + str('%.4f' % (rec_mean/all_n)) + ' ' + str('%.4f' % (f1_mean/all_n)))
        f.write('\n')
        f.write('混淆矩阵： ')
        f.write('\n')
        for i in range(11):
            for j in range(11):
                f.write(str(mix_array[i][j]) + ' ')
            f.write('\n')

    print(mix_array)


if __name__ == '__main__':
    # 转换标注json 160 to 10
    tranform_160_to_10()

    # 获取mask
    make_mask(label_json_path, label_mask_path, aliyun_name_to_value)
    make_mask(predict_json_path, predict_mask_path, damage_name_to_value_24)

    # 计算
    count_aliyun(damage_name_to_value_24)


