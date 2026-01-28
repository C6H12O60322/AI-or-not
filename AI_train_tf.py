import os
import io
import json
import tqdm
import shutil
import pprint
import pathlib
import tempfile
import requests
import collections
import matplotlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from six import BytesIO
from etils import epath
from IPython import display
from urllib.request import urlopen

import orbit
import tensorflow as tf
import tensorflow_models as tfm
import tensorflow_datasets as tfds

from official.core import exp_factory
from official.core import config_definitions as cfg
from official.vision.data import tfrecord_lib
from official.vision.serving import export_saved_model_lib
from official.vision.dataloaders.tf_example_decoder import TfExampleDecoder
from official.vision.utils.object_detection import visualization_utils
from official.vision.ops.preprocess_ops import normalize_image, resize_and_crop_image
from official.vision.data.create_coco_tf_record import coco_annotations_to_lists

pp = pprint.PrettyPrinter(indent=4) # Set Pretty Print Indentation
print(tf.__version__) # Check the version of tensorflow used

_URLS = {
    'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
    'validation_images': 'http://images.cocodataset.org/zips/val2017.zip',
    'test_images': 'http://images.cocodataset.org/zips/test2017.zip',
}

train_prefix = 'train'
valid_prefix = 'val'

train_annotation_path = './lvis_v1_train.json'
valid_annotation_path = './lvis_v1_val.json'

IMGS_DIR = './lvis_sub_dataset/'
tf_records_dir = './lvis_tfrecords/'


if not os.path.exists(IMGS_DIR):
  os.mkdir(IMGS_DIR)

if not os.path.exists(tf_records_dir):
  os.mkdir(tf_records_dir)



NUM_CLASSES = 3
category_index = get_category_map(valid_annotation_path, NUM_CLASSES)
category_ids = list(category_index.keys())

# Below helper function are taken from github tensorflow dataset lvis
# https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/datasets/lvis/lvis_dataset_builder.py
_generate_tf_records(train_prefix,
                     _URLS['train_images'],
                     train_annotation_path)

_generate_tf_records(valid_prefix,
                     _URLS['validation_images'],
                     valid_annotation_path)
#為自訂資料集配置 MaskRCNN Resnet FPN COCO 模型

train_data_input_path = './lvis_tfrecords/train*'
valid_data_input_path = './lvis_tfrecords/val*'
test_data_input_path = './lvis_tfrecords/test*'
model_dir = './trained_model/'
export_dir ='./exported_model/'

if not os.path.exists(model_dir):
  os.mkdir(model_dir)
'''在 Model Garden 中，定義模型的參數集合稱為configs。Model Garden 可以透過工廠根據一組已知參數建立配置。

使用retinanet_mobilenet_coco實驗配置，如 所定義tfm.vision.configs.maskrcnn.maskrcnn_mobilenet_coco。

請在這裡找到所有已註冊的實驗

此配置定義了一個實驗，用於訓練以 mobilenet 作為主幹、FPN 作為解碼器的 Mask R-CNN 模型。預設配置在COCO train2017上進行訓練並在COCO val2017 上進行評估。

還有其他可供選擇的實驗，例如 maskrcnn_resnetfpn_coco， maskrcnn_spinenet_coco等等。人們可以透過更改函數的實驗名稱參數來切換到它們get_exp_config。
'''

exp_config = exp_factory.get_exp_config('maskrcnn_mobilenet_coco')

model_ckpt_path = './model_ckpt/'
if not os.path.exists(model_ckpt_path):
  os.mkdir(model_ckpt_path)

!gsutil cp gs://tf_model_garden/vision/mobilenet/v2_1.0_float/ckpt-180648.data-00000-of-00001 './model_ckpt/'
!gsutil cp gs://tf_model_garden/vision/mobilenet/v2_1.0_float/ckpt-180648.index './model_ckpt/'
#調整模型和資料集配置，使其適用於自訂資料集。

BATCH_SIZE = 8
HEIGHT, WIDTH = 256, 256
IMG_SHAPE = [HEIGHT, WIDTH, 3]


# Backbone Config
exp_config.task.annotation_file = None
exp_config.task.freeze_backbone = True
exp_config.task.init_checkpoint = "./model_ckpt/ckpt-180648"
exp_config.task.init_checkpoint_modules = "backbone"

# Model Config
exp_config.task.model.num_classes = NUM_CLASSES + 1
exp_config.task.model.input_size = IMG_SHAPE

# Training Data Config
exp_config.task.train_data.input_path = train_data_input_path
exp_config.task.train_data.dtype = 'float32'
exp_config.task.train_data.global_batch_size = BATCH_SIZE
exp_config.task.train_data.shuffle_buffer_size = 64
exp_config.task.train_data.parser.aug_scale_max = 1.0
exp_config.task.train_data.parser.aug_scale_min = 1.0

# Validation Data Config
exp_config.task.validation_data.input_path = valid_data_input_path
exp_config.task.validation_data.dtype = 'float32'
exp_config.task.validation_data.global_batch_size = BATCH_SIZE
#調整訓練器配置。

logical_device_names = [logical_device.name for logical_device in tf.config.list_logical_devices()]

if 'GPU' in ''.join(logical_device_names):
  print('This may be broken in Colab.')
  device = 'GPU'
elif 'TPU' in ''.join(logical_device_names):
  print('This may be broken in Colab.')
  device = 'TPU'
else:
  print('Running on CPU is slow, so only train for a few steps.')
  device = 'CPU'


train_steps = 2000
exp_config.trainer.steps_per_loop = 200 # steps_per_loop = num_of_training_examples // train_batch_size

exp_config.trainer.summary_interval = 200
exp_config.trainer.checkpoint_interval = 200
exp_config.trainer.validation_interval = 200
exp_config.trainer.validation_steps =  200 # validation_steps = num_of_validation_examples // eval_batch_size
exp_config.trainer.train_steps = train_steps
exp_config.trainer.optimizer_config.warmup.linear.warmup_steps = 200
exp_config.trainer.optimizer_config.learning_rate.type = 'cosine'
exp_config.trainer.optimizer_config.learning_rate.cosine.decay_steps = train_steps
exp_config.trainer.optimizer_config.learning_rate.cosine.initial_learning_rate = 0.07
exp_config.trainer.optimizer_config.warmup.linear.warmup_learning_rate = 0.05
#列印修改後的配置。

pp.pprint(exp_config.as_dict())
display.Javascript("google.colab.output.setIframeHeight('500px');")
#制定分銷策略。

# Setting up the Strategy
if exp_config.runtime.mixed_precision_dtype == tf.float16:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

if 'GPU' in ''.join(logical_device_names):
  distribution_strategy = tf.distribute.MirroredStrategy()
elif 'TPU' in ''.join(logical_device_names):
  tf.tpu.experimental.initialize_tpu_system()
  tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='/device:TPU_SYSTEM:0')
  distribution_strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
  print('Warning: this will be really slow.')
  distribution_strategy = tf.distribute.OneDeviceStrategy(logical_device_names[0])

print("Done")
'''
從建立Task物件 ( ) 。tfm.core.base_task.Taskconfig_definitions.TaskConfig
該Task物件具有建構資料集、建置模型以及運行訓練和評估所需的所有方法。這些方法是由 驅動的tfm.core.train_lib.run_experiment。
'''

with distribution_strategy.scope():
  task = tfm.core.task_factory.get_task(exp_config.task, logging_dir=model_dir)
#可視化一批數據。

for images, labels in task.build_inputs(exp_config.task.train_data).take(1):
  print()
  print(f'images.shape: {str(images.shape):16}  images.dtype: {images.dtype!r}')
  print(f'labels.keys: {labels.keys()}')
#建立類別索引字典以將標籤對應到對應的標籤名稱

tf_ex_decoder = TfExampleDecoder(include_mask=True)
'''用於視覺化 TFRecords 結果的輔助函數
使用visualize_boxes_and_labels_on_image_arrayfromvisualization_utils在影像上繪製邊界框。
'''

def show_batch(raw_records):
  plt.figure(figsize=(20, 20))
  use_normalized_coordinates=True
  min_score_thresh = 0.30
  for i, serialized_example in enumerate(raw_records):
    plt.subplot(1, 3, i + 1)
    decoded_tensors = tf_ex_decoder.decode(serialized_example)
    image = decoded_tensors['image'].numpy().astype('uint8')
    scores = np.ones(shape=(len(decoded_tensors['groundtruth_boxes'])))
    # print(decoded_tensors['groundtruth_instance_masks'].numpy().shape)
    # print(decoded_tensors.keys())
    visualization_utils.visualize_boxes_and_labels_on_image_array(
        image,
        decoded_tensors['groundtruth_boxes'].numpy(),
        decoded_tensors['groundtruth_classes'].numpy().astype('int'),
        scores,
        category_index=category_index,
        use_normalized_coordinates=use_normalized_coordinates,
        min_score_thresh=min_score_thresh,
        instance_masks=decoded_tensors['groundtruth_instance_masks'].numpy().astype('uint8'),
        line_thickness=4)

    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Image-{i+1}")
  plt.show()
'''
列車數據視覺化
邊界框偵測由三個部分組成

偵測到的物件的類別標籤。
預測邊界框與真實邊界框之間的匹配百分比。
實例分割掩碼
注意：一切都是 100% 的原因是因為我們正在視覺化真實情況
'''
buffer_size = 100
num_of_examples = 3

train_tfrecords = tf.io.gfile.glob(exp_config.task.train_data.input_path)
raw_records = tf.data.TFRecordDataset(train_tfrecords).shuffle(buffer_size=buffer_size).take(num_of_examples)
show_batch(raw_records)
'''訓練和評估
我們遵循COCO挑戰傳統，基於mAP（平均精度）來評估目標偵測的準確性。請在此處查看有關如何完成檢測任務的評估指標的詳細說明。

IoU：定義為交集面積除以預測邊界框和地面真實邊界框並集的面積。
'''

model, eval_logs = tfm.core.train_lib.run_experiment(
    distribution_strategy=distribution_strategy,
    task=task,
    mode='train_and_eval',
    params=exp_config,
    model_dir=model_dir,
    run_post_eval=True)
#在張量板上載入日誌

%load_ext tensorboard
%tensorboard --logdir "./trained_model"
'''儲存並匯出訓練後的模型
keras.Model傳回的物件期望train_lib.run_experiment資料集載入器使用 中相同的平均值和變異數統計資料對資料進行標準化preprocess_ops.normalize_image(image, offset=MEAN_RGB, scale=STDDEV_RGB)。此導出函數處理這些細節，因此您可以傳遞tf.uint8圖像並獲得正確的結果。
'''

export_saved_model_lib.export_inference_graph(
    input_type='image_tensor',
    batch_size=1,
    input_image_size=[HEIGHT, WIDTH],
    params=exp_config,
    checkpoint_path=tf.train.latest_checkpoint(model_dir),
    export_dir=export_dir)
#從訓練模型推斷

def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: the file path to the image

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  image = None
  if(path.startswith('http')):
    response = urlopen(path)
    image_data = response.read()
    image_data = BytesIO(image_data)
    image = Image.open(image_data)
  else:
    image_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(image_data))

  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (1, im_height, im_width, 3)).astype(np.uint8)



def build_inputs_for_object_detection(image, input_image_size):
  """Builds Object Detection model inputs for serving."""
  image, _ = resize_and_crop_image(
      image,
      input_image_size,
      padded_size=input_image_size,
      aug_scale_min=1.0,
      aug_scale_max=1.0)
  return image
#視覺化測試數據

num_of_examples = 3

test_tfrecords = tf.io.gfile.glob('./lvis_tfrecords/val*')
test_ds = tf.data.TFRecordDataset(test_tfrecords).take(num_of_examples)
show_batch(test_ds)
#導入已儲存的模型

imported = tf.saved_model.load(export_dir)
model_fn = imported.signatures['serving_default']
#視覺化預測

def reframe_image_corners_relative_to_boxes(boxes):
  """Reframe the image corners ([0, 0, 1, 1]) to be relative to boxes.
  The local coordinate frame of each box is assumed to be relative to
  its own for corners.
  Args:
    boxes: A float tensor of [num_boxes, 4] of (ymin, xmin, ymax, xmax)
      coordinates in relative coordinate space of each bounding box.
  Returns:
    reframed_boxes: Reframes boxes with same shape as input.
  """
  ymin, xmin, ymax, xmax = (boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3])

  height = tf.maximum(ymax - ymin, 1e-4)
  width = tf.maximum(xmax - xmin, 1e-4)

  ymin_out = (0 - ymin) / height
  xmin_out = (0 - xmin) / width
  ymax_out = (1 - ymin) / height
  xmax_out = (1 - xmin) / width
  return tf.stack([ymin_out, xmin_out, ymax_out, xmax_out], axis=1)

def reframe_box_masks_to_image_masks(box_masks, boxes, image_height,
                                     image_width, resize_method='bilinear'):
  """Transforms the box masks back to full image masks.
  Embeds masks in bounding boxes of larger masks whose shapes correspond to
  image shape.
  Args:
    box_masks: A tensor of size [num_masks, mask_height, mask_width].
    boxes: A tf.float32 tensor of size [num_masks, 4] containing the box
           corners. Row i contains [ymin, xmin, ymax, xmax] of the box
           corresponding to mask i. Note that the box corners are in
           normalized coordinates.
    image_height: Image height. The output mask will have the same height as
                  the image height.
    image_width: Image width. The output mask will have the same width as the
                 image width.
    resize_method: The resize method, either 'bilinear' or 'nearest'. Note that
      'bilinear' is only respected if box_masks is a float.
  Returns:
    A tensor of size [num_masks, image_height, image_width] with the same dtype
    as `box_masks`.
  """
  resize_method = 'nearest' if box_masks.dtype == tf.uint8 else resize_method
  # TODO(rathodv): Make this a public function.
  def reframe_box_masks_to_image_masks_default():
    """The default function when there are more than 0 box masks."""

    num_boxes = tf.shape(box_masks)[0]
    box_masks_expanded = tf.expand_dims(box_masks, axis=3)

    resized_crops = tf.image.crop_and_resize(
        image=box_masks_expanded,
        boxes=reframe_image_corners_relative_to_boxes(boxes),
        box_indices=tf.range(num_boxes),
        crop_size=[image_height, image_width],
        method=resize_method,
        extrapolation_value=0)
    return tf.cast(resized_crops, box_masks.dtype)

  image_masks = tf.cond(
      tf.shape(box_masks)[0] > 0,
      reframe_box_masks_to_image_masks_default,
      lambda: tf.zeros([0, image_height, image_width, 1], box_masks.dtype))
  return tf.squeeze(image_masks, axis=3)

input_image_size = (HEIGHT, WIDTH)
plt.figure(figsize=(20, 20))
min_score_thresh = 0.40 # Change minimum score for threshold to see all bounding boxes confidences

for i, serialized_example in enumerate(test_ds):
  plt.subplot(1, 3, i+1)
  decoded_tensors = tf_ex_decoder.decode(serialized_example)
  image = build_inputs_for_object_detection(decoded_tensors['image'], input_image_size)
  image = tf.expand_dims(image, axis=0)
  image = tf.cast(image, dtype = tf.uint8)
  image_np = image[0].numpy()
  result = model_fn(image)
  # Visualize detection and masks
  if 'detection_masks' in result:
    # we need to convert np.arrays to tensors
    detection_masks = tf.convert_to_tensor(result['detection_masks'][0])
    detection_boxes = tf.convert_to_tensor(result['detection_boxes'][0])
    detection_masks_reframed = reframe_box_masks_to_image_masks(
              detection_masks, detection_boxes/256.0,
                image_np.shape[0], image_np.shape[1])
    detection_masks_reframed = tf.cast(
        detection_masks_reframed > min_score_thresh,
        np.uint8)

    result['detection_masks_reframed'] = detection_masks_reframed.numpy()
  visualization_utils.visualize_boxes_and_labels_on_image_array(
        image_np,
        result['detection_boxes'][0].numpy(),
        (result['detection_classes'][0] + 0).numpy().astype(int),
        result['detection_scores'][0].numpy(),
        category_index=category_index,
        use_normalized_coordinates=False,
        max_boxes_to_draw=200,
        min_score_thresh=min_score_thresh,
        instance_masks=result.get('detection_masks_reframed', None),
        line_thickness=4)

  plt.imshow(image_np)
  plt.axis("off")

plt.show()