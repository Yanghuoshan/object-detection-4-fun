import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from utils import transform
from lxml import etree
from xml.etree import ElementTree

"""
VOC2007数据集格式:
└──VOCdevkit
    └──VOC2007
        └──JPEGImages
            └──0.jpg
            └──1.jpg
            └──2.jpg
            └──...
        └──Annotations
            └──0.xml
            └──1.xml
            └──2.xml
            └──...
        └──ImageSets
            └──Main
                └──train.txt
                └──val.txt
                └──trainval.txt
                └──test.txt

"""

'''
xml文件信息(例):
<annotation>
    <folder>JPEGImages</folder>
    <filename>0.jpg</filename>
    <path>X:/.../.../VOCdevkit/VOC2007/JPEGImages/0.jpg</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>600</width>
        <height>800</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>peach</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>100</xmin>
            <ymin>250</ymin>
            <xmax>150</xmax>
            <ymax>260</ymax>
        </bndbox>
    </object>
    <object>
        <name>cat</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>200</xmin>
            <ymin>50</ymin>
            <xmax>550</xmax>
            <ymax>370</ymax>
        </bndbox>
    </object>
</annotation>
'''


class CustomDataset(Dataset):  # 自定义数据集
    def __init__(self, root, transforms=None, dataset_property="train_list"):  # 初始化方法
        self.root = root  # 数据路径，应指向".../.../VOCdevkit"
        self.transforms = transforms  # 预处理方法，一般来说需要传入，注意区分训练数据和验证数据的预处理方法
        self.images_dir = os.path.join(self.root, "JPEGImages")  # 图像文件存储路径，默认VOC2007/JPEGImages
        self.annotations_dir = os.path.join(self.root, "Annotations")  # 标注文件存储路径，默认VOC2007/Annotations
        self.imagesets_dir = os.path.join(self.root, "Main")  # 数据集划分文件存储路径，VOC2007/ImageSets/Main
        self.dataset_property=dataset_property

        assert dataset_property in ["train_list", "val_list", "trainval_list", "test_list"]  # 数据集划分文件名应该为train/val/trainval/test，以txt形式存储
        with open(os.path.join(self.imagesets_dir, f"{dataset_property}.txt")) as f:  # 打开对应txt文件
            self.data_names = [i.split()[0][11:-4] for i in f.readlines()]  # 读取数据，存放于列表(每张图片的名称，不包含.jpg后缀)

        # print(self.data_names)

        # self.label_dict = {  # 语义标签与int的对应关系，一般从1开始，0表示背景(一般通过读取json或txt文件来获得对应关系)
        #     "cat": 1,
        #     "dog": 2,
        #     "peach": 3
        # }

        self.label_dict = {'3+2-2': 1, '3jia2': 2, 'aerbeisi': 3, 'anmuxi': 4, 'aoliao': 5, 'asamu': 6, 'baicha': 7,
                           'baishikele': 8, 'baishikele-2': 9, 'baokuangli': 10, 'binghongcha': 11,
                           'bingqilinniunai': 12, 'bingtangxueli': 13, 'buding': 14, 'chacui': 15, 'chapai': 16,
                           'chapai2': 17, 'damaicha': 18, 'daofandian1': 19, 'daofandian2': 20, 'daofandian3': 21,
                           'daofandian4': 22, 'dongpeng': 23, 'dongpeng-b': 24, 'fenda': 25, 'gudasao': 26,
                           'guolicheng': 27, 'guolicheng2': 28, 'haitai': 29, 'haochidian': 30, 'haoliyou': 31,
                           'heweidao': 32, 'heweidao2': 33, 'heweidao3': 34, 'hongniu': 35, 'hongniu2': 36,
                           'hongshaoniurou': 37, 'jianjiao': 38, 'jianlibao': 39, 'jindian': 40, 'kafei': 41,
                           'kaomo_gali': 42, 'kaomo_jiaoyan': 43, 'kaomo_shaokao': 44, 'kaomo_xiangcon': 45,
                           'kebike': 46, 'kele': 47, 'kele-b': 48, 'kele-b-2': 49, 'laotansuancai': 50,
                           'liaomian': 51, 'libaojian': 52, 'lingdukele': 53, 'lingdukele-b': 54, 'liziyuan': 55,
                           'lujiaoxiang': 56, 'lujikafei': 57, 'luxiangniurou': 58, 'maidong': 59,
                           'mangguoxiaolao': 60, 'meiniye': 61, 'mengniu': 62, 'mengniuzaocan': 63, 'moliqingcha': 64,
                           'nfc': 65, 'niudufen': 66, 'niunai': 67, 'nongfushanquan': 68, 'qingdaowangzi-1': 69,
                           'qingdaowangzi-2': 70, 'qinningshui': 71, 'quchenshixiangcao': 72, 'rancha-1': 73,
                           'rancha-2': 74, 'rousongbing': 75, 'rusuanjunqishui': 76, 'suanlafen': 77,
                           'suanlaniurou': 78, 'taipingshuda': 79, 'tangdaren': 80, 'tangdaren2': 81, 'tangdaren3': 82,
                           'ufo': 83, 'ufo2': 84, 'wanglaoji': 85, 'wanglaoji-c': 86, 'wangzainiunai': 87, 'weic': 88,
                           'weitanai': 89, 'weitanai2': 90, 'weitanaiditang': 91, 'weitaningmeng': 92,
                           'weitaningmeng-bottle': 93, 'weiweidounai': 94, 'wuhounaicha': 95, 'wulongcha': 96,
                           'xianglaniurou': 97, 'xianguolao': 98, 'xianxiayuban': 99, 'xuebi': 100, 'xuebi-b': 101,
                           'xuebi2': 102, 'yezhi': 103, 'yibao': 104, 'yida': 105, 'yingyangkuaixian': 106,
                           'yitengyuan': 107, 'youlemei': 108, 'yousuanru': 109, 'youyanggudong': 110,
                           'yuanqishui': 111, 'zaocanmofang': 112, 'zihaiguo': 113}

    def __len__(self):  # 获取数据集长度方法  **该方法必须定义**
        return len(self.data_names)  # 返回数据列表的长度

    def __getitem__(self, index):  # 采样方法，根据索引index取得对应的图像image和标签target  **该方法必须定义**
        image_path = os.path.join(self.images_dir, self.data_names[index] + ".jpg")  # 图片路径，要求图片为jpg格式
        image = Image.open(image_path).convert("RGB")  # PIL图像

        annotation_path = os.path.join(self.annotations_dir, self.data_names[index] + ".xml")  # xml路径
        label_names, difficults, boxes, areas = self.read_xml(annotation_path)  # 解析xml文件，获取标注信息：标签名称、难识别目标标签、边界框、面积
        labels = [self.label_dict[f"{label}"] for label in label_names]  # 按照对应关系，将语义标签转化为int类型

        labels, difficults, boxes, areas = map(lambda t: torch.as_tensor(t),
                                               [labels, difficults, boxes, areas])  # 将标注信息转化为tensor形式

        # target = {  # 构建最后返回的target，一般还包括image_id(图像id)、masks(分割掩码)、iscrowd(是否为多目标)
        #     "boxes": boxes,  # 边界框
        #     "labels": labels,  # 标签
        #     "area": areas,  # 面积
        #     "isdifficult": difficults  # 难识别目标标签
        # }

        # if self.transforms is not None:  # 如果使用预处理方法
        #     image, target = self.transforms(image, target)  # 进行image和target的预处理，transforms函数/类需要复写，以满足对target的变换

        image, boxes, labels, difficulties = transform(image, boxes, labels, difficults, split=self.dataset_property)

        return image, boxes, labels, difficults  # 返回image，target

    def read_xml(self, annotation_path):  # 读取xml信息方法
        objnames = []  # 用于存放目标名称
        difficults = []  # 用于存放难识别目标标签
        objboxes = []  # 用于存放边界框
        objareas = []  # 用于存放面积

        parser = etree.XMLParser(encoding="utf-8")  # xml文件解析器
        xmlroot = ElementTree.parse(annotation_path, parser=parser).getroot()  # 解析xml文件并获得root节点
        for object in xmlroot.findall("object"):  # 寻找所有object节点并遍历
            objnames.append(object.find("name").text)  # 获取name节点数据，填入列表
            difficults.append(int(object.find("difficult").text))  # 获得difficult节点数据，转换为int，填入列表
            objxmin = float(object.find("bndbox/xmin").text)  # 获得bndbox/xmin节点数据，转换为float
            objymin = float(object.find("bndbox/ymin").text)  # 获得bndbox/ymin节点数据，转换为float
            objxmax = float(object.find("bndbox/xmax").text)  # 获得bndbox/xmax节点数据，转换为float
            objymax = float(object.find("bndbox/ymax").text)  # 获得bndbox/ymax节点数据，转换为float
            assert objxmax > objxmin and objymax > objymin  # 检查边界框的长宽是否为正
            objboxes.append([objxmin, objymin, objxmax, objymax])  # 边界框[xmin,ymin,xmax,ymax]，填入列表
            objareas.append((objxmax - objxmin) * (objymax - objymin))  # 面积，填入列表

        return objnames, difficults, objboxes, objareas  # 返回目标名称、难识别目标标签、边界框、面积

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each
