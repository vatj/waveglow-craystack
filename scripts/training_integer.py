#!/usr/bin/env python
# coding: utf-8

# # Train WaveGlow Model with custom training step

# ## Boilerplate Import

# In[ ]:


import os, sys
root_dir, _ = os.path.split(os.getcwd())
script_dir = os.path.join(root_dir, 'scripts')
sys.path.append(script_dir)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


# In[ ]:


import tensorflow as tf


# In[ ]:


from hparams import hparams
from waveglow_model_integer import WaveGlowInteger
import training_utils as utils
from waveglow_codec import compress_samples
from datetime import datetime


# In[ ]:


tf.keras.backend.clear_session()
tf.config.experimental.list_physical_devices('GPU')
tf.keras.backend.set_floatx()
# tf.debugging.set_log_device_placement(True)


# ## Tensorboard logs setup

# In[ ]:


log_dir = os.path.join(hparams['log_dir'])
file_writer = tf.summary.create_file_writer(log_dir)
file_writer.set_as_default()


# ## Load Validation and Training Dataset

# In[ ]:


validation_dataset = utils.load_single_file_tfrecords(
  record_file=os.path.join(hparams['tfrecords_dir'], hparams['eval_file']))
validation_dataset = validation_dataset.batch(
  hparams['train_batch_size'])
validation_dataset = validation_dataset.unbatch().batch(hparams['compress_batch_size'])


# In[ ]:


training_dataset = utils.load_training_files_tfrecords(
  record_pattern=os.path.join(hparams['tfrecords_dir'], hparams['train_files'] + '*'))


# ## Instantiate model and optimizer

# In[ ]:


myWaveGlow = WaveGlowInteger(hparams=hparams, name='myWaveGlow')
optimizer = utils.get_optimizer(hparams=hparams)


# ## Model Checkpoints : Initialise or Restore

# In[ ]:


checkpoint = tf.train.Checkpoint(step=tf.Variable(0), 
                                 optimizer=optimizer, 
                                 net=myWaveGlow)

manager_checkpoint = tf.train.CheckpointManager(
  checkpoint, 
  directory=hparams['checkpoint_dir'],
  max_to_keep=hparams['max_to_keep'])

checkpoint.restore(manager_checkpoint.latest_checkpoint)

if manager_checkpoint.latest_checkpoint:
  tf.summary.experimental.set_step(tf.cast(checkpoint.step, tf.int64))
  tf.summary.text(name="checkpoint_restore",
                  data="Restored from {}".format(manager_checkpoint.latest_checkpoint))
else:
  tf.summary.experimental.set_step(0)


# ## Training step autograph

# In[ ]:


@tf.function
def train_step(step, x_train, waveGlow, hparams, optimizer):
  tf.summary.experimental.set_step(step=step)
  with tf.GradientTape() as tape:
    outputs = waveGlow(x_train, training=True)
    total_loss = waveGlow.total_loss(outputs=outputs)

  grads = tape.gradient(total_loss, 
                        waveGlow.trainable_weights)
  optimizer.apply_gradients(zip(grads, 
                                waveGlow.trainable_weights))


# In[ ]:


def custom_training(waveGlow, hparams, optimizer, 
                    checkpoint, manager_checkpoint):
  step = tf.cast(checkpoint.step, tf.int64)
  
  for epoch in tf.range(hparams['epochs']):
    tf.summary.text(name='epoch',
                    data='Start epoch {}'.format(epoch.numpy()) +\
                    'at ' + datetime.now().strftime("%Y%m%d-%H%M%S"),
                    step=step)
    
    for step, x_train in training_dataset.enumerate(start=step):
      train_step(step=step,
                 x_train=x_train,
                 waveGlow=waveGlow,
                 hparams=hparams,
                 optimizer=optimizer)
      
      if tf.equal(step % hparams['save_model_every'], 0):
        save_path = manager_checkpoint.save()
        tf.summary.text(name='save_checkpoint',
                        data="Saved checkpoint in" + save_path,
                        step=step)

      if tf.equal(step > tf.constant(25000, dtype=tf.int64), True):
        save_compression_every = 25000
      elif tf.equal(step > tf.constant(5000, dtype=tf.int64), True):
        save_compression_every = 5000
      else:
        save_compression_every = hparams['save_compression_every']

      if tf.equal(step % save_compression_every, 0):
        tf.print(f'Compression test {step}')
        compress_samples(model=waveGlow, hparams=hparams, step=step, decode=False)
        tf.print('End test')
    
      checkpoint.step.assign_add(1)


# In[ ]:


custom_training(waveGlow=myWaveGlow, 
                hparams=hparams, 
                optimizer=optimizer,
                checkpoint=checkpoint,
                manager_checkpoint=manager_checkpoint)


# In[ ]:




