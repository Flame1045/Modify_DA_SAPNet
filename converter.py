import json
import os
import os.path
from PIL import Image

def position(pos):
    # 该函数用来找出xmin,ymin,xmax,ymax即bbox包围框
    x = []
    y = []
    nums = len(pos)
    for i in range(nums):
        x.append(pos[i][0])
        y.append(pos[i][1])
    x_max = max(x)
    x_min = min(x)
    y_max = max(y)
    y_min = min(y)
    b = (float(x_min), float(y_min), float(x_max), float(y_max))
    return b

def convert_annotation(image_id):

        load_f = open(rootdir + '/' + image_id + "_gtFine_polygons.json", 'r')  # 导入json标签的路径
        load_dict = json.load(load_f)
        out_file = open(rootdir + '/' + '%s_leftImg8bit.txt' % (image_id), 'w')  # 输出标签的路径
        objects = load_dict['objects']
        nums = len(objects)

        cls_id = ''
        for i in range(0, nums):
            labels = objects[i]['label']
            if (labels in ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']): #这里我需要用到的类别是这8类
                print(labels)
                pos = objects[i]['polygon']
                bb = position(pos)
                cls_id = labels
                out_file.write(cls_id + " " + " ".join([str(a) for a in bb]) + '\n')
        if cls_id == '':
            print('no label json:', "%s_gtFine_polygons.json" % (image_id))


def images_id(orgin_picture_dir):  # 获取训练集每个图像的名称  （orgin_picture_dir训练集图像的路径）
    a = []
    for parent, dirnames, filenames in os.walk(orgin_picture_dir):
        for filename in filenames:
            print(filename)
            filename = filename.split('_leftImg8bit.png')[0]
            print(filename)
            a.append(filename)
    return a


def xml_generator(orgin_picture_dir, rootdir): #生成xml文件
    img_basenames = os.listdir(orgin_picture_dir)
    img_names = []
    for item in img_basenames:
        temp1, temp2 = os.path.splitext(item)  # 分别提取图片名称和图片后缀名称（具有_leftImg8bit）
        img_names.append(temp1)
        print(img_names)
    for img in img_names:  # img是没有后缀的pic名称
        im = Image.open((orgin_picture_dir + '/' +img + '.png'))
        width, height = im.size

        # open the crospronding txt file
        gt = open(rootdir + '/' + img + '.txt').read().splitlines()

        # write in xml file
        xml_file = open((rootdir + '/' + img + '.xml'), 'w')
        xml_file.write('<annotation>\n')
        xml_file.write('    <folder>CITYSCAPE</folder>\n')
        xml_file.write('    <filename>' + str(img) + '.png' + '</filename>\n')
        xml_file.write('    <size>\n')
        xml_file.write('        <width>' + str(width) + '</width>\n')
        xml_file.write('        <height>' + str(height) + '</height>\n')
        xml_file.write('        <depth>3</depth>\n')
        xml_file.write('    </size>\n')

        # write the region of image on xml file
        for img_each_label in gt:
            spt = img_each_label.split(' ')  # 这里如果txt里面是以逗号‘，’隔开的，那么就改为spt = img_each_label.split(',')。
            xml_file.write('    <object>\n')
            xml_file.write('        <name>' + str(spt[0]) + '</name>\n')
            xml_file.write('        <pose>Unspecified</pose>\n')
            xml_file.write('        <truncated>0</truncated>\n')
            xml_file.write('        <difficult>0</difficult>\n')
            xml_file.write('        <bndbox>\n')
            xml_file.write('            <xmin>' + str(spt[1]) + '</xmin>\n')
            xml_file.write('            <ymin>' + str(spt[2]) + '</ymin>\n')
            xml_file.write('            <xmax>' + str(spt[3]) + '</xmax>\n')
            xml_file.write('            <ymax>' + str(spt[4]) + '</ymax>\n')
            xml_file.write('        </bndbox>\n')
            xml_file.write('    </object>\n')
        xml_file.write('</annotation>')

if __name__ == '__main__':
    city_name = os.listdir('/home/eric/Desktop/research/M10902116/SAPNetV2/datasets/Cityscapes-coco/gtFine/test/')
    print(city_name)
    for city in city_name:
              rootdir = '/home/eric/Desktop/research/M10902116/SAPNetV2/datasets/Cityscapes-coco/gtFine/test/' + city   # json所在路径
              orgin_picture_dir = '/home/eric/Desktop/research/M10902116/SAPNetV2/datasets/Cityscapes-coco/leftImg8bit/test/' + city    #cityscape的train所在路径
              names = images_id(orgin_picture_dir)
              for image_id in names:
                print(image_id)
                convert_annotation(image_id)
              xml_generator(orgin_picture_dir, rootdir)
