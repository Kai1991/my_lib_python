import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('json_input', '', 'Path to the json input')
flags.DEFINE_string('image_dir', '', 'Path to the image directory')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


# M1，this code part need to be modified according to your real situation
def class_text_to_int(row_label):
    if row_label == "破洞":
        return 1
    elif row_label == "水渍":
        return 2
    elif row_label == "油渍":
        return 2
    elif row_label == "污渍":
        return 2
    elif row_label == "三丝":
        return 3
    elif row_label == "结头":
        return 4
    elif row_label == "花板跳":
        return 5
    elif row_label == "百脚":
        return 6
    elif row_label == "毛粒":
        return 7
    elif row_label == "粗经":
        return 8
    elif row_label == "松经":
        return 9
    elif row_label == "断经":
        return 10
    elif row_label == "吊经":
        return 11
    elif row_label == "粗维":
        return 12
    elif row_label == "纬缩":
        return 13
    elif row_label == "浆斑":
        return 14
    elif row_label == "整经结":
        return 15
    elif row_label == "星跳":
        return 16
    elif row_label == "跳花":
        return 16
    elif row_label == "断氨纶":
        return 17
    elif row_label == "稀密档":
        return 18
    elif row_label == "浪纹档":
        return 18
    elif row_label == "色差档":
        return 18
    elif row_label == "磨痕":
        return 19
    elif row_label == "轧痕":
        return 19
    elif row_label == "修痕":
        return 19
    elif row_label == "烧毛痕":
        return 19
    elif row_label == "死皱":
        return 20
    elif row_label == "云织":
        return 20
    elif row_label == "双纬":
        return 20
    elif row_label == "双经":
        return 20
    elif row_label == "跳纱":
        return 20
    elif row_label == "筘路":
        return 20
    elif row_label == "纬纱不良":
        return 20
    else:
        None


def load_data(path="/Volumes/移动硬盘/data/guangdong1_round1_train1_20190818/Annotations/anno_train.json",print_counter=True):
    data = namedtuple('data', ['filename', 'object'])
    json_datas = json.load(open(path,'r'))
    datas = dict()
    counter = Counter()
    for json_data in json_datas:
        if json_data['name'] in datas.keys():
            datas.get(json_data['name']).append(json_data)
        else:
            datas[json_data['name']] = [json_data]
        counter.update([json_data['defect_name']])
    if print_counter:
        log(counter)
    return [data(name, obj) for name, obj in zip(datas.keys(), datas.values())]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['bbox'][0] / width)
        xmaxs.append(row['bbox'][2] / width)
        ymins.append(row['bbox'][1] / height)
        ymaxs.append(row['bbox'][3] / height)
        classes_text.append(row['defect_name'].encode('utf8'))
        classes.append(class_text_to_int(row['defect_name']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(os.getcwd(), FLAGS.image_dir)
    examples = json.load(open(FLAGS.json_input,'r'))
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()