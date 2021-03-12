import numpy as np


# 读取txt文件
def read_txt(txt_path):
    filenames = []
    with open(txt_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            filenames.append(line)
    return filenames


# 划分统一的测试数据集
def split_unified_test():
    train_160_txt = "C:\\Users\\Administrator\\Desktop\\统一测试集\\160分类-train.txt"
    val_160_txt = "C:\\Users\\Administrator\\Desktop\\统一测试集\\160分类-val.txt"
    test_160_txt = "C:\\Users\\Administrator\\Desktop\\统一测试集\\160分类-test.txt"

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
    for filename in train_160_filenames:
        if filename in unified_test_filenames:
            cnt += 1

    print("cnt:", cnt)






if __name__ == '__main__':
    split_unified_test()