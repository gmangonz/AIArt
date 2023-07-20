import tensorflow as tf
import os
from PIL import Image
from tqdm import tqdm
from glob import glob

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_tf_example(image_path):
    """Creates an Example proto for the given image.
    Args:
      image_path: Path to the image file.

    Returns:
      tf_example: A tf.train.Example
    """
    # Get bytes of image
    with tf.io.gfile.GFile(image_path, 'rb') as f:
      try:
        encoded_image_data = f.read()
      except:
        print(f'{image_path} gives Error')

    # Get image width and height
    image = Image.open(image_path)
    width, height = image.size

    # Create tf.train.Example
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/encoded': bytes_feature(encoded_image_data),
    }))

    return tf_example

def create_tf_records(image_dir, output_path, start_index = 0):

    if not os.path.isfile(output_path):
      
      print(f'Saving output to: {output_path}')

      images = sorted(glob(os.path.join(image_dir, '*')), key=os.path.getsize)[start_index:]
      expected_size = sum(list(map(lambda x: os.path.getsize(x), images))) / (1e+9)

      print(f'Writing {len(images)} to TFRecord file, expected size (GB) is {expected_size} ...')

      writer = tf.io.TFRecordWriter(output_path)

      for image_path in tqdm(images):

          tf_example = create_tf_example(image_path)
          writer.write(tf_example.SerializeToString())

      writer.close()

      print("TFRecord files created successfully!")
    else:
      print(f'Skipping, {output_path} aleady exists')


if __name__ == '__main__':
  datapath = r'/content/drive/MyDrive/Rice Undergrad/Film 499 - AI Art/DataSet'
  sub_folders = list(map(lambda x: os.path.join(datapath, 'Abstract_Imgs', x), os.listdir(os.path.join(datapath, 'Abstract_Imgs'))))
  for folder in sub_folders:
    output = os.path.join('/content/drive/MyDrive/DL-CV-ML Projects/All_Data/AIART', os.path.basename(folder))
    create_tf_records(folder, output)

