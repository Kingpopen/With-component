import numpy as np
import os
import json
import traceback
import pandas as pd
import random

# 读取txt文件
def read_txt(txt_path):
    filenames = []
    with open(txt_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip("\n")
            filenames.append(line)
    return filenames

# 保存txt文件
def save_txt(txt_path, filenames):
    with open(txt_path, 'w') as f:
        for filename in filenames:
            f.write(filename + "\n")

component_label_160_name_to_value = {"_background_":0,
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

damage_label_160_name_to_value = {
    "_background_": 0,
    "scratch": 1,
    "indentation": 2,
    "crack": 3,
    "perforation": 4,
    "serious": 5
}


# 划分统一的测试数据集
def split_unified_test():
    train_160_txt = "C:\\Users\\Administrator\\Desktop\\train.txt"
    val_160_txt = "C:\\Users\\Administrator\\Desktop\\统一测试集\\160分类-val.txt"
    test_160_txt = "C:\\Users\\Administrator\\Desktop\\test.txt"

    train_52_txt = "C:\\Users\\Administrator\\Desktop\\统一测试集\\52分类-train.txt"
    val_52_txt = "C:\\Users\\Administrator\\Desktop\\统一测试集\\52分类-val.txt"
    test_52_txt = "C:\\Users\\Administrator\\Desktop\\统一测试集\\52分类-test.txt"

    train_24_txt = "C:\\Users\\Administrator\\Desktop\\统一测试集\\24分类-train.txt"
    val_24_txt = "C:\\Users\\Administrator\\Desktop\\统一测试集\\24分类-val.txt"
    test_24_txt = "C:\\Users\\Administrator\\Desktop\\统一测试集\\24分类-test.txt"

    unified_test_txt = "C:\\Users\\Administrator\\Desktop\\统一测试集\\unified-test.txt"

    train_160_filenames = read_txt(train_160_txt)
    val_160_filenames = read_txt(val_160_txt)
    test_160_filenames = read_txt(test_160_txt)

    train_52_filenames = read_txt(train_52_txt)
    val_52_filenames = read_txt(val_52_txt)
    test_52_filenames = read_txt(test_52_txt)

    train_24_filenames = read_txt(train_24_txt)
    val_24_filenames = read_txt(val_24_txt)
    test_24_filenames = read_txt(test_24_txt)

    unified_test_filenames = read_txt(unified_test_txt)

    cnt = 0
    final_filenames = []


    for filename in train_160_filenames:
        if filename in unified_test_filenames:
            cnt += 1
            final_filenames.append(filename)
    print("cnt:", cnt)

    cnt = 0
    print("the len of unified_test_filenames:", len(unified_test_filenames))
    print("the len of test_160_filenames:", len(test_160_filenames))

    for filename in unified_test_filenames:
        if filename in test_160_filenames:
            cnt += 1
    print("cnt:", cnt)


# 简化划分数据集的方法
'''
思路：
直接给160类数据集划分三大类 训练集，验证集，测试集
然后从上面三类中选出符合阿里、百度的数据集
'''
def split_dataset_simple():
    origin_jsonfile_path = "C:\\Users\\Administrator\\Desktop\\0111-损伤_17504_json(原始)"
    total_txt_160_path = "C:\\Users\\Administrator\\Desktop\\test_txt\\total.txt"
    total_160_filenames = read_txt(total_txt_160_path)

    # 将数据集打乱
    random.shuffle(total_160_filenames)
    # 数据集数目获取
    train_160_num = int(len(total_160_filenames) * 0.8)
    test_160_num = int(len(total_160_filenames) * 0.1)
    val_160_num = len(total_160_filenames) - train_160_num - test_160_num

    train_160_filenames = total_160_filenames[0: train_160_num]
    val_160_filenames = total_160_filenames[train_160_num: train_160_num+val_160_num]
    test_160_filenames = total_160_filenames[train_160_num + val_160_num:]

    # 通过160分类的数据集 获取阿里云数据集的训练集、验证集、测试集
    train_24_filenames, val_24_filenames, test_24_filenames = get_24_dataset_txt(origin_jsonfile_path, train_160_filenames, val_160_filenames, test_160_filenames)
    train_52_filenames, val_52_filenames, test_52_filenames = get_52_dataset_txt(origin_jsonfile_path, train_160_filenames, val_160_filenames, test_160_filenames)

    print("len of test_160:", len(test_160_filenames))
    print("len of test_24:", len(test_24_filenames))
    print("len of test_52:", len(test_52_filenames))

    # 进行保存
    save_train_24_txt = "C:\\Users\\Administrator\\Desktop\\0318-simple_split\\new_24分类_train.txt"
    save_val_24_txt = "C:\\Users\\Administrator\\Desktop\\0318-simple_split\\new_24分类_val.txt"
    save_test_24_txt = "C:\\Users\\Administrator\\Desktop\\0318-simple_split\\new_24分类_test.txt"

    save_train_52_txt = "C:\\Users\\Administrator\\Desktop\\0318-simple_split\\new_52分类_train.txt"
    save_val_52_txt = "C:\\Users\\Administrator\\Desktop\\0318-simple_split\\new_52分类_val.txt"
    save_test_52_txt = "C:\\Users\\Administrator\\Desktop\\0318-simple_split\\new_52分类_test.txt"

    save_train_160_txt = "C:\\Users\\Administrator\\Desktop\\0318-simple_split\\new_160分类_train.txt"
    save_val_160_txt = "C:\\Users\\Administrator\\Desktop\\0318-simple_split\\new_160分类_val.txt"
    save_test_160_txt = "C:\\Users\\Administrator\\Desktop\\0318-simple_split\\new_160分类_test.txt"

    save_txt(save_train_24_txt, train_24_filenames)
    save_txt(save_train_52_txt, train_52_filenames)
    save_txt(save_train_160_txt, train_160_filenames)

    save_txt(save_val_24_txt, val_24_filenames)
    save_txt(save_val_52_txt, val_52_filenames)
    save_txt(save_val_160_txt, val_160_filenames)

    save_txt(save_test_24_txt, test_24_filenames)
    save_txt(save_test_52_txt, test_52_filenames)
    save_txt(save_test_160_txt, test_160_filenames)


# 获取24分类的训练集、验证集、测试集
def get_24_dataset_txt(origin_jsonfile_path, trainnames, valnames, testnames):
    result_trainnames, result_valnames, result_testnames= [], [], []
    for filename in trainnames:
        jsonpath = os.path.join(origin_jsonfile_path, filename + ".json")
        data = json.load(open(jsonpath))
        shapes = data["shapes"]
        flag = False
        for shape in shapes:
            group_id = shape["group_id"]
            # 将严重损伤排除掉
            if group_id != 4:
                flag = True
        if flag:
            result_trainnames.append(filename)
    for filename in valnames:
        jsonpath = os.path.join(origin_jsonfile_path, filename + ".json")
        data = json.load(open(jsonpath))
        shapes = data["shapes"]
        flag = False
        for shape in shapes:
            group_id = shape["group_id"]
            # 将严重损伤排除掉
            if group_id != 4:
                flag = True
        if flag:
            result_valnames.append(filename)

    for filename in testnames:
        jsonpath = os.path.join(origin_jsonfile_path, filename + ".json")
        data = json.load(open(jsonpath))
        shapes = data["shapes"]
        flag = False
        for shape in shapes:
            group_id = shape["group_id"]
            # 将严重损伤排除掉
            if group_id != 4:
                flag = True
        if flag:
            result_testnames.append(filename)


    return result_trainnames, result_valnames, result_testnames

# 获取52分类的训练集、验证集、测试集
def get_52_dataset_txt(origin_jsonfile_path, trainnames, valnames, testnames):

    result_trainnames, result_valnames, result_testnames = [], [], []
    for filename in trainnames:
        jsonpath = os.path.join(origin_jsonfile_path, filename + ".json")
        data = json.load(open(jsonpath))
        shapes = data["shapes"]

        flag = False
        for shape in shapes:
            group_id = shape["group_id"]
            label = shape["label"]
            # 将严重损伤排除掉
            if group_id != 4:
                if label in ["front door", "rear door"]:
                    label = "door"
                elif label in ["rear outer taillight", "rear inner taillight"]:
                    label = "rear taillight"

                if label in ["front bumper", "rear bumper", "front fender", "rear fender", "door", "rear taillight", "headlight",
                             "hood", "luggage cover", "radiator grille", "bottom side", "rearview mirror","license plate"]:
                    flag = True

        if flag:
            result_trainnames.append(filename)
    for filename in valnames:
        jsonpath = os.path.join(origin_jsonfile_path, filename + ".json")
        data = json.load(open(jsonpath))
        shapes = data["shapes"]

        flag = False
        for shape in shapes:
            group_id = shape["group_id"]
            label = shape["label"]
            # 将严重损伤排除掉
            if group_id != 4:
                if label in ["front door", "rear door"]:
                    label = "door"
                elif label in ["rear outer taillight", "rear inner taillight"]:
                    label = "rear taillight"

                if label in ["front bumper", "rear bumper", "front fender", "rear fender", "door", "rear taillight",
                             "headlight",
                             "hood", "luggage cover", "radiator grille", "bottom side", "rearview mirror",
                             "license plate"]:
                    flag = True

        if flag:
            result_valnames.append(filename)
    for filename in testnames:
        jsonpath = os.path.join(origin_jsonfile_path, filename + ".json")
        data = json.load(open(jsonpath))
        shapes = data["shapes"]

        flag = False
        for shape in shapes:
            group_id = shape["group_id"]
            label = shape["label"]
            # 将严重损伤排除掉
            if group_id != 4:
                if label in ["front door", "rear door"]:
                    label = "door"
                elif label in ["rear outer taillight", "rear inner taillight"]:
                    label = "rear taillight"

                if label in ["front bumper", "rear bumper", "front fender", "rear fender", "door", "rear taillight",
                             "headlight",
                             "hood", "luggage cover", "radiator grille", "bottom side", "rearview mirror",
                             "license plate"]:
                    flag = True

        if flag:
            result_testnames.append(filename)

    return result_trainnames, result_valnames, result_testnames

# 检查3
def check():
    train_160_txt = "C:\\Users\\Administrator\\Desktop\\0318-simple_split\\new_160分类_train.txt"
    val_160_txt = "C:\\Users\\Administrator\\Desktop\\0318-simple_split\\new_160分类_val.txt"
    test_160_txt = "C:\\Users\\Administrator\\Desktop\\0318-simple_split\\new_160分类_test.txt"

    train_52_txt = "C:\\Users\\Administrator\\Desktop\\0318-simple_split\\new_52分类_train.txt"
    val_52_txt = "C:\\Users\\Administrator\\Desktop\\0318-simple_split\\new_52分类_val.txt"
    test_52_txt = "C:\\Users\\Administrator\\Desktop\\0318-simple_split\\new_52分类_test.txt"

    train_24_txt = "C:\\Users\\Administrator\\Desktop\\0318-simple_split\\new_24分类_train.txt"
    val_24_txt = "C:\\Users\\Administrator\\Desktop\\0318-simple_split\\new_24分类_val.txt"
    test_24_txt = "C:\\Users\\Administrator\\Desktop\\0318-simple_split\\new_24分类_test.txt"

    train_24_filenames = read_txt(train_24_txt)
    val_24_filenames = read_txt(val_24_txt)
    test_24_filenames = read_txt(test_24_txt)

    train_52_filenames = read_txt(train_52_txt)
    val_52_filenames = read_txt(val_52_txt)
    test_52_filenames = read_txt(test_52_txt)

    train_160_filenames = read_txt(train_160_txt)
    val_160_filenames = read_txt(val_160_txt)
    test_160_filenames = read_txt(test_160_txt)

    cnt = 0
    for filename in train_160_filenames:
        if (filename in train_24_filenames) and (filename in train_52_filenames):
            cnt += 1
    print("repeat train:", cnt)

    cnt = 0
    for filename in val_160_filenames:
        if (filename in val_24_filenames) and (filename in val_52_filenames):
            cnt += 1
    print("repeat val:", cnt)

    cnt = 0
    for filename in test_160_filenames:
        if (filename in test_24_filenames) and (filename in test_52_filenames):
            cnt += 1
    print("repeat test:", cnt)

class Statistic_instance():
    def __init__(self):
        train_160_txt = "C:\\Users\\Administrator\\Desktop\\0318-simple_split\\new_160分类_train.txt"
        val_160_txt = "C:\\Users\\Administrator\\Desktop\\0318-simple_split\\new_160分类_val.txt"
        test_160_txt = "C:\\Users\\Administrator\\Desktop\\0318-simple_split\\new_160分类_test.txt"

        train_52_txt = "C:\\Users\\Administrator\\Desktop\\0318-simple_split\\new_52分类_train.txt"
        val_52_txt = "C:\\Users\\Administrator\\Desktop\\0318-simple_split\\new_52分类_val.txt"
        test_52_txt = "C:\\Users\\Administrator\\Desktop\\0318-simple_split\\new_52分类_test.txt"

        train_24_txt = "C:\\Users\\Administrator\\Desktop\\0318-simple_split\\new_24分类_train.txt"
        val_24_txt = "C:\\Users\\Administrator\\Desktop\\0318-simple_split\\new_24分类_val.txt"
        test_24_txt = "C:\\Users\\Administrator\\Desktop\\0318-simple_split\\new_24分类_test.txt"
        origin_jsonfile_path = "C:\\Users\\Administrator\\Desktop\\0111-损伤_17504_json(原始)"


# 统计各种数据集中instance的种类个数
def statistic_instance():
    train_160_txt = "C:\\Users\\Administrator\\Desktop\\0319-统一数据集\\new_160分类_train.txt"
    val_160_txt = "C:\\Users\\Administrator\\Desktop\\0319-统一数据集\\new_160分类_val.txt"
    test_160_txt = "C:\\Users\\Administrator\\Desktop\\0319-统一数据集\\new_160分类_test.txt"

    train_52_txt = "C:\\Users\\Administrator\\Desktop\\0319-统一数据集\\new_52分类_train.txt"
    val_52_txt = "C:\\Users\\Administrator\\Desktop\\0319-统一数据集\\new_52分类_val.txt"
    test_52_txt = "C:\\Users\\Administrator\\Desktop\\0319-统一数据集\\new_52分类_test.txt"

    train_24_txt = "C:\\Users\\Administrator\\Desktop\\0319-统一数据集\\new_52分类_train.txt"
    val_24_txt = "C:\\Users\\Administrator\\Desktop\\0319-统一数据集\\new_52分类_val.txt"
    test_24_txt = "C:\\Users\\Administrator\\Desktop\\0319-统一数据集\\new_52分类_test.txt"

    origin_jsonfile_path = "C:\\Users\\Administrator\\Desktop\\0111-损伤_17504_json(原始)"
    save_xlsx_path = "C:\\Users\\Administrator\\Desktop\\0319-统一数据集"

    train_24_filenames = read_txt(train_24_txt)
    val_24_filenames = read_txt(val_24_txt)
    test_24_filenames = read_txt(test_24_txt)

    train_52_filenames = read_txt(train_52_txt)
    val_52_filenames = read_txt(val_52_txt)
    test_52_filenames = read_txt(test_52_txt)

    train_160_filenames = read_txt(train_160_txt)
    val_160_filenames = read_txt(val_160_txt)
    test_160_filenames = read_txt(test_160_txt)

    # 统计160分类instance数目(返回的结果为一个二维数组)
    instances_train_160 = statistic_160_instance(origin_jsonfile_path, train_160_filenames)
    print("instances_train_160 over")
    instances_val_160 = statistic_160_instance(origin_jsonfile_path, val_160_filenames)
    print("instances_val_160 over")
    instances_test_160 = statistic_160_instance(origin_jsonfile_path, test_160_filenames)
    print("instances_test_160 over")

    # 统计24分类instance数目
    instances_train_24 = statistic_24_instance(origin_jsonfile_path, train_24_filenames)
    print("instances_train_24 over")
    instances_val_24 = statistic_24_instance(origin_jsonfile_path, val_24_filenames)
    print("instances_val_24 over")
    instances_test_24 = statistic_24_instance(origin_jsonfile_path, test_24_filenames)
    print("instances_test_24 over")

    # 统计52分类instance数目
    instances_train_52 = statistic_52_instance(origin_jsonfile_path, train_52_filenames)
    print("instances_train_52 over")
    instances_val_52 = statistic_52_instance(origin_jsonfile_path, val_52_filenames)
    print("instances_val_52 over")
    instances_test_52 = statistic_52_instance(origin_jsonfile_path, test_52_filenames)
    print("instances_test_52 over")

    # 统计10分类的instances的数目
    instances_train_10_dict = statistic_10_instance(origin_jsonfile_path, train_24_filenames)
    print("instances_train_10 over")
    instances_val_10_dict = statistic_10_instance(origin_jsonfile_path, val_24_filenames)
    print("instances_val_10 over")
    instances_test_10_dict = statistic_10_instance(origin_jsonfile_path, test_24_filenames)
    print("instances_test_10 over")


    # save
    save_160_instance(instances_test_160, os.path.join(save_xlsx_path, "./instance_test_160.xlsx"))
    save_160_instance(instances_train_160, os.path.join(save_xlsx_path, "./instance_train_160.xlsx"))
    save_160_instance(instances_val_160, os.path.join(save_xlsx_path, "./instance_val_160.xlsx"))

    save_52_instance(instances_test_52, os.path.join(save_xlsx_path, "./instance_test_52.xlsx"))
    save_52_instance(instances_train_52, os.path.join(save_xlsx_path, "./instance_train_52.xlsx"))
    save_52_instance(instances_val_52, os.path.join(save_xlsx_path, "./instance_val_52.xlsx"))
    #
    save_24_instance(instances_test_24, os.path.join(save_xlsx_path, "./instance_test_24.xlsx"))
    save_24_instance(instances_train_24, os.path.join(save_xlsx_path, "./instance_train_24.xlsx"))
    save_24_instance(instances_val_24, os.path.join(save_xlsx_path, "./instance_val_24.xlsx"))

    print("instances_train_10_dict:", instances_train_10_dict)
    print("instances_val_10_dict:", instances_val_10_dict)
    print("instances_test_10_dict:", instances_test_10_dict)


# 统计160分类的结果
def statistic_160_instance(origin_jsonfile_path, filenames):
    '''

    '''
    component_label_160_name_to_value = {"front bumper": 0, "rear bumper": 1,
                                         "front bumper grille": 2, "front windshield": 3,
                                         "rear windshield": 4, "front tire": 5,
                                         "rear tire": 6, "front side glass": 7,
                                         "rear side glass": 8, "front fender": 9,
                                         "rear fender": 10, "front mudguard": 11,
                                         "rear mudguard": 12, "turn signal": 13,
                                         "front door": 14, "rear door": 15,
                                         "rear outer taillight": 16, "rear inner taillight": 17,
                                         "headlight": 18, "fog light": 19,
                                         "hood": 20, "luggage cover": 21,
                                         "roof": 22, "steel ring": 23,
                                         "radiator grille": 24, "a pillar": 25,
                                         "b pillar": 26, "c pillar": 27,
                                         "d pillar": 28, "bottom side": 29,
                                         "rearview mirror": 30, "license plate": 31}

    instances_160 = np.zeros((32, 5))
    cnt = 0
    try:
        for filename in filenames:
            jsonpath = os.path.join(origin_jsonfile_path, filename + ".json")
            data = json.load(open(jsonpath))
            shapes = data["shapes"]
            cnt += 1
            for shape in shapes:
                label = shape["label"]
                group_id = shape["group_id"]
                instances_160[component_label_160_name_to_value[label], group_id] += 1
            print("cnt:", cnt)

    except KeyError:
        print("filename:", filename)
        print("label:", label)
        print("group_id:", group_id)
        traceback.print_exc()

    return instances_160
# 统计阿里10分类的结果
def statistic_10_instance(origin_jsonfile_path ,filenames):
    '''
    origin_file_path is a path to store json files
    filenames is a list of filename like [16543..., 1324324...,...]
    return the dict of  10 classes instances
    '''
    dataset_10_dict = {
        "scratch": 0,
        "indentation": 0,
        "crack and perforation": 0,
        "gap": 0,
        "light scratch": 0,
        "light perforation": 0,
        "rearview scratch": 0,
        "rearview perforation": 0,
        "rearview crack": 0,
        "windshield perforation": 0
    }
    cnt = 0
    for filename in filenames:
        jsonpath = os.path.join(origin_jsonfile_path, filename + ".json")
        data = json.load(open(jsonpath))
        shapes = data["shapes"]

        cnt += 1

        for shape in shapes:
            label = shape["label"]
            group_id = shape["group_id"]
            try:
                # scratch：刮擦  (除了大灯，后视镜)
                if (group_id == 0) and \
                        (label not in ["headlight", "rearview mirror", "rear outer taillight",
                                       "rear inner taillight"]):
                    label = "scratch"
                    dataset_10_dict[label] += 1

                # indentation：变形（所有零件）
                elif group_id == 1:
                    label = "indentation"
                    dataset_10_dict[label] += 1

                # crack and perforation：破损孔洞（除了前后翼子板和前后保险杠，后视镜，前后挡风玻璃，前大灯，后内外尾灯）
                elif (group_id == 2 or group_id == 3) and \
                        (label not in ["front fender", "rear fender", "front bumper", "rear bumper",
                                       "rearview mirror",
                                       "headlight", "rear outer taillight", "rear inner taillight",
                                       "front windshield",
                                       "rear windshield"]):
                    label = "crack and perforation"
                    dataset_10_dict[label] += 1

                # gap：缝隙（前后保险杠和前后翼子板的开裂）
                elif (group_id == 2) and (label in ["front bumper", "rear bumper", "front fender", "rear fender"]):
                    label = "gap"
                    dataset_10_dict[label] += 1

                # light scratch：大灯刮擦（前大灯  后内外尾灯）
                elif (group_id == 0) and (label in ["headlight", "rear outer taillight", "rear inner taillight"]):
                    label = "light scratch"
                    dataset_10_dict[label] += 1


                # light perforation：大灯破损（前大灯  后内外尾灯）
                elif (group_id == 3) and (label in ["headlight", "rear outer taillight", "rear inner taillight"]):
                    label = "light perforation"
                    dataset_10_dict[label] += 1

                # rearview scratch：后视镜刮擦（后视镜）
                elif (group_id == 0) and (label == "rearview mirror"):
                    label = "rearview scratch"
                    dataset_10_dict[label] += 1

                # rearview perforation：后视镜破损（后视镜）
                elif (group_id == 3) and (label == "rearview mirror"):
                    label = "rearview perforation"
                    dataset_10_dict[label] += 1

                # rearview crack：后视镜脱落（后视镜）
                elif (group_id == 2) and (label == "rearview mirror"):
                    label = "rearview crack"
                    dataset_10_dict[label] += 1

                # windshield perforation：挡风玻璃破损（前后挡风玻璃）
                elif (group_id == 3) and (label in ["front windshield", "rear windshield"]):
                    label = "windshield perforation"
                    dataset_10_dict[label] += 1
            except KeyError:
                print("filename:", filename)
                traceback.print_exc()
        print("cnt:", cnt)
    return dataset_10_dict

# 统计阿里24分类的结果（多分枝）
def statistic_24_instance(origin_jsonfile_path, filenames):
    '''

    '''
    component_label_24_name_to_value = {"bumper": 0,
                                        "fender": 1,
                                        "light": 2,
                                        "rearview": 3,
                                        "windshield": 4,
                                        "others": 5
                                         }

    instances_24 = np.zeros((6, 4))
    cnt = 0
    for filename in filenames:
        jsonpath = os.path.join(origin_jsonfile_path, filename + ".json")
        data = json.load(open(jsonpath))
        shapes = data["shapes"]
        cnt += 1
        for shape in shapes:
            label = shape["label"]
            group_id = shape["group_id"]
            # 将严重损伤排除掉
            if group_id != 4:
                if label in ["front bumper", "rear bumper"]:
                    label = "bumper"
                elif label in ["front fender", "rear fender"]:
                    label = "fender"
                elif label in ["headlight", "rear outer taillight", "rear inner taillight"]:
                    label = "light"
                elif label == "rearview mirror":
                    label = "rearview"
                elif label in ["front windshield", "rear windshield"]:
                    label = "windshield"
                else:
                    label = "others"


                instances_24[component_label_24_name_to_value[label], group_id] += 1
            print("cnt:", cnt)
    return instances_24
# 统计百度52分类的结果
def statistic_52_instance(origin_jsonfile_path, filenames):
    '''

    '''
    component_label_52_name_to_value ={
        "front bumper":0, "rear bumper":1, "front fender":2, "rear fender":3, "door":4, "rear taillight":5,
        "headlight":6, "hood":7, "luggage cover":8, "radiator grille":9, "bottom side":10, "rearview mirror":11,
        "license plate":12
    }
    cnt = 0
    instances_52 = np.zeros((13, 4))
    for filename in filenames:
        jsonpath = os.path.join(origin_jsonfile_path, filename + ".json")
        data = json.load(open(jsonpath))
        shapes = data["shapes"]

        cnt += 1

        for shape in shapes:
            group_id = shape["group_id"]
            label = shape["label"]
            # 将严重损伤排除掉
            if group_id != 4:
                if label in ["front door", "rear door"]:
                    label = "door"
                elif label in ["rear outer taillight", "rear inner taillight"]:
                    label = "rear taillight"

                if label in ["front bumper", "rear bumper", "front fender", "rear fender", "door", "rear taillight", "headlight",
                             "hood", "luggage cover", "radiator grille", "bottom side", "rearview mirror","license plate"]:
                    instances_52[component_label_52_name_to_value[label], group_id] += 1

            print("cnt:", cnt)
    return instances_52

# 保存160类的instances各类别数目
def save_160_instance(instance_array, save_path):
    instance_df = pd.DataFrame(instance_array)
    instance_df.columns = ["scratch", "indentation", "crack", "perforation", "serious"]
    instance_df.index = ["front bumper", "rear bumper", "front bumper grille", "front windshield",
                            "rear windshield", "front tire", "rear tire", "front side glass", "rear side glass",
                            "front fender", "rear fender", "front mudguard","rear mudguard", "turn signal",
                            "front door", "rear door", "rear outer taillight", "rear inner taillight",
                            "headlight", "fog light", "hood", "luggage cover", "roof", "steel ring",
                            "radiator grille", "a pillar", "b pillar", "c pillar", "d pillar", "bottom side",
                            "rearview mirror", "license plate"]
    writer = pd.ExcelWriter(save_path)
    instance_df.to_excel(writer, 'Pagee')
    writer.save()

# 保存52类的instances各类别数目
def save_52_instance(instance_array, save_path):
    instance_df = pd.DataFrame(instance_array)
    instance_df.columns = ["scratch", "indentation", "crack", "perforation"]
    instance_df.index = [ "front bumper", "rear bumper", "front fender", "rear fender", "door", "rear taillight",
        "headlight", "hood", "luggage cover", "radiator grille", "bottom side", "rearview mirror","license plate"]
    writer = pd.ExcelWriter(save_path)
    instance_df.to_excel(writer, 'Pagee')
    writer.save()

# 保存24类的instances各类别数目
def save_24_instance(instance_array, save_path):
    instance_df = pd.DataFrame(instance_array)
    instance_df.columns = ["scratch", "indentation", "crack", "perforation"]
    instance_df.index = ["bumper", "fender", "light", "rearview", "windshield", "others"]
    writer = pd.ExcelWriter(save_path)
    instance_df.to_excel(writer, 'Pagee')
    writer.save()

# 保存10类的instances各类别数目
def save_10_instance(instance_dict, save_path):
    pass


# 将原始的json转换为多分支的阿里格式
def json2alijson_multi(data):
    pass

# 将原始的json转换为多分支的百度格式
def json2baidujson_multi():
    pass



def fan1():
    train_24_txt = "C:\\Users\\Administrator\\Desktop\\0318-simple_split\\new_24分类_train.txt"
    val_24_txt = "C:\\Users\\Administrator\\Desktop\\0318-simple_split\\new_24分类_val.txt"
    test_24_txt = "C:\\Users\\Administrator\\Desktop\\0318-simple_split\\new_24分类_test.txt"

    train_52_txt = "C:\\Users\\Administrator\\Desktop\\0318-simple_split\\new_52分类_train.txt"
    val_52_txt = "C:\\Users\\Administrator\\Desktop\\0318-simple_split\\new_52分类_val.txt"
    test_52_txt = "C:\\Users\\Administrator\\Desktop\\0318-simple_split\\new_52分类_test.txt"

    train_24_filenames = read_txt(train_24_txt)
    val_24_filenames = read_txt(val_24_txt)
    test_24_filenames = read_txt(test_24_txt)

    train_52_filenames = read_txt(train_52_txt)
    val_52_filenames = read_txt(val_52_txt)
    test_52_filenames = read_txt(test_52_txt)

    common_train_filenames = []
    cnt = 0
    for filename in train_24_filenames:
        if filename in train_52_filenames:
           cnt += 1

    print("common train:", cnt)

    cnt = 0
    for filename in val_24_filenames:
        if filename in val_52_filenames:
           cnt += 1
    print("common val:", cnt)

    cnt = 0
    for filename in test_24_filenames:
        if filename in test_52_filenames:
            cnt += 1
    print("common test:", cnt)



if __name__ == '__main__':
    statistic_instance()
    # split_dataset_simple()
    # check()
    # fan1()