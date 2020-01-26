#!/usr/bin/env python
# coding: utf-8

# # Useful functions (needs refactoring)

# In[28]:


import os, sys
root_dir, _ = os.path.split(os.getcwd())
script_dir = os.path.join(root_dir, 'scripts')
sys.path.append(script_dir)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# In[31]:


import tensorflow as tf
import numpy as np
import pprint


# In[ ]:


from hparams import hparams


# ## Dataset loading

# In[ ]:


sound_feature_description = {
    "wav": tf.io.FixedLenFeature([], tf.string)
}

def _parse_sound_function(example_proto):
  x = tf.io.parse_single_example(example_proto, sound_feature_description)
  x['wav'] = tf.io.parse_tensor(x['wav'], out_type=hparams['ftype']) 
  return x

long_sound_feature_description = {
  "wav": tf.io.FixedLenFeature([], tf.string),
  "path": tf.io.FixedLenFeature([], tf.string),
  "number_of_slices": tf.io.FixedLenFeature([], tf.string)
}

def _parse_long_sound_function(example_proto):
  x = tf.io.parse_single_example(example_proto, long_sound_feature_description)
  x['wav'] = tf.io.parse_tensor(x['wav'], out_type=hparams['ftype'])
  x['path'] = tf.io.parse_tensor(x['path'], out_type=tf.string)
  x['number_of_slices'] = tf.io.parse_tensor(x['number_of_slices'], out_type=tf.int32)  
  return x


# In[ ]:


def load_single_file_tfrecords(record_file):
  raw_sound_dataset = tf.data.TFRecordDataset(record_file)
  parsed_sound_dataset = raw_sound_dataset.map(_parse_sound_function)
  return parsed_sound_dataset

def load_long_audio_tfrecords(record_file):
  raw_sound_dataset = tf.data.TFRecordDataset(record_file)
  parsed_sound_dataset = raw_sound_dataset.map(_parse_long_sound_function)
  return parsed_sound_dataset

def load_training_files_tfrecords(record_pattern):
  record_files = tf.data.TFRecordDataset.list_files(
    file_pattern=record_pattern)
  raw_sound_dataset = record_files.interleave(
    tf.data.TFRecordDataset,
    cycle_length=1,
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
  parsed_sound_dataset = raw_sound_dataset.map(
    _parse_sound_function,
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
  
  training_dataset = parsed_sound_dataset.shuffle(
    buffer_size=hparams['buffer_size']).batch(
    hparams['train_batch_size'],
    drop_remainder=True).prefetch(
    buffer_size=tf.data.experimental.AUTOTUNE)
  
  return training_dataset


# ## Optimizer compatibility with tf.float16 (Not working yet)

# In[ ]:


def get_optimizer(hparams):
  """
  Return optimizer instance based on hparams
  
  Wrap the optimizer to avoid underflow if ftype=tf.float16
  """
  if hparams['optimizer'] == "Adam":
    if hparams['learning_rate_decay']:
      lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        hparams['learning_rate'],
        decay_steps=200000,
        decay_rate=0.99,
        staircase=True)
      optimizer = tf.keras.optimizers.Adam(
      learning_rate=lr_schedule)
    else:
      optimizer = tf.keras.optimizers.Adam(
        learning_rate=hparams["learning_rate"])

  elif hparams['optimizer'] == "Adadelta":
    assert(hparams["learning_rate"] == 1.0), "Set learning_rate to 1.0"
    optimizer = tf.keras.optimizers.Adadelta(
      learning_rate=hparams['learning_rate'])
  else:
    raise ValueError("Supported Optimizer is either Adam or Adadelta")
    
  if hparams["mixed_precision"]:
    return tf.train.experimental.enable_mixed_precision_graph_rewrite(
      optimizer, "dynamic")
  else:
    return optimizer


# ## Discrete Logistic Mixture Helpers

# In[ ]:


def log_sum_exp(x):
  """ numerically stable log_sum_exp implementation that prevents overflow """
  axis = len(x.shape)-1
  m = tf.reduce_max(x, axis)
  m2 = tf.reduce_max(x, axis, keepdims=True)
  return m + tf.math.log(tf.reduce_sum(tf.math.exp(x-m2), axis))

def log_prob_from_logits(x):
  """ numerically stable log_softmax implementation that prevents overflow """
  axis = len(x.shape)-1
  m = tf.reduce_max(x, axis, keepdims=True)
  return x - m - tf.math.log(tf.reduce_sum(tf.math.exp(x-m), axis, keepdims=True))


# ## Straight-through estimator

# In[ ]:


@tf.custom_gradient
def round_ste(x):
  def grad(dy):
    return dy
  return tf.round(x), grad


# ## Integer values

# In[ ]:




