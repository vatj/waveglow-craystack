#!/usr/bin/env python
# coding: utf-8

# # Hyperparameter dictionary

# In[ ]:


import os, sys


# In[ ]:


import tensorflow as tf


# In[ ]:


hparams = dict()


# ## Waveglow parameters

# In[ ]:


# Number of flow blocks/composite layers in the model. Each flow is Invertible Convolution + Integer Coupling Layer to Wavenet
hparams['n_flows'] = 16
# Number of initial channels in invertible convolution
hparams['n_group'] = 16
# Number of flow layers between each channel reduction
hparams['n_early_every'] = 4
# Number of channels to shave from audio sample every n_early_every
hparams['n_early_size'] = 4
# Hidden Channels
hparams["hidden_channels"] = 256
# Number of Dlogistic
hparams['n_logistic_in_mixture'] = 10
# Last log_scale 
hparams['last_log_scale'] = -6.5
# Seed to set permutation layers
hparams['seed'] = 235


# ## Wavenet Parameters

# In[ ]:


# Number of composite layer in Wavenet acting first with separate Conv1D on audio and spectrogram
# followed by sigmoid and tanh activations. Last comes a skip_layer
hparams['n_layers'] = 8
# Kernel size of the Wavenet convolution, no causality restriction
hparams['kernel_size'] = 3
# Number of hidden channels in Wavenet composite layers
hparams['n_channels'] = 256
# Only half of the audio sample channels go through wavenet.
hparams['n_in_channels'] = 8


# ## Preprocessing of audio samples from LJSpeech

# In[ ]:


# Crop audio samples to length pow(2, _) . Small samples of pow(2, _) (14 is about 10% of original sample length)
hparams['segment_length'] = pow(2, 10)
# Wav file sampling rate
hparams['sample_rate'] = 22050


# ## Machinery Details

# In[ ]:


# Floating precision. 
hparams['ftype'] = tf.float32
# Batch size for training
hparams['train_batch_size'] = 24
# Learning rate, set to range(1e-3, 1e-4) for Adam and 1.0 for AdaDelta. Learning rate scheduler not supported yet
hparams['learning_rate'] = 1e-4
# Number of epochs to iterate over. Might be replaced by a number of training step in the future
hparams['epochs'] = 1000
# Buffer size for shuffling
hparams['buffer_size'] = 240
# Optimizer, either Adam or AdaDelta. If AdaDelta, learning_rate = 1.0. 
# Added experimental tensorflow wrapper to support mixed precision and tf.float16
hparams['optimizer'] = "Adam" 
# Enable mixed precision calculation. 
hparams['mixed_precision'] = False # Not fully implemented yet.
# Enable learning rate decay
hparams["learning_rate_decay"] = False # Not implemented yet.


# In[ ]:


# Save model every number of step
hparams['save_model_every'] = 1000
# Save audio samples every number of step
hparams['save_audio_every'] = 5000
# Try compression every number of step
hparams['save_compression_every'] = 5000
# Number of checkpoint files to keep
hparams['max_to_keep'] = 3 


# ## Craystack Parameters

# In[ ]:


# batch_size for compression tasks
hparams['compress_batch_size'] = 1
# zdim for craystack
hparams['zdim'] = (hparams['segment_length'] // 4, 1)
hparams['xdim'] = (hparams['segment_length'], 1)
# Coding precision of the encoder. Value above 28 produce warning but necessary to rebalance buckets
hparams['coding_prec'] = 31 
# Bin precision for LogisticMixture_UnifBins codec. Necessary 16 since encoding np.int16
hparams['bin_precision'] = 16 
# Lower bound for LogisticMixture_UnifBins codec
hparams['bin_lower_bound'] = -1.
# Upper bound for LogisticMixture_UnifBins codec
hparams['bin_upper_bound'] = 1.
# Number of datapoint to compress. Higher value allows to offset the overhead associated to rANS encoder.
# 10 samples for segment_length of pow(2, 14) takes roughly 1 hour on my cpu and ~16GB RAM
hparams['n_compress_datapoint'] = 10


# ## Generate tfrecords files

# In[ ]:


# Split training data in n_shards tfrecords files
hparams['n_shards'] = 12 
# Number of excluded example from training set to generate audio samples at each epochs for quality assessment
hparams['n_eval_samples'] = 20 
# Number of excluded audio samples to run test on after training
hparams['n_test_samples'] = 80
# Training tfrecords common filename
hparams['train_files'] = 'ljs_train'
# Evaluation tfrecords filename
hparams['eval_file'] = 'ljs_eval.tfrecords'
# Test tfrecords filename
hparams['test_file'] = 'ljs_test.tfrecords'
# Long audio sample filename
hparams['long_audio_file'] = 'ljs_long.tfrecords'


# ## Path

# In[ ]:


# Common file name
save_name = '{segment_length}c_craystack_seed={seed}_flows={n_flows}'.format(**hparams)
# Machine Home Path
hparams['base_path'] = os.path.join('/', 'home', 'victor', 'Projects', 'Github')
# Raw data directory
hparams['data_dir'] = os.path.join('/', 'home', 'victor', '.keras', 'datasets', 'LJSpeech-1.1')
# Tfrecords directory.
hparams['tfrecords_dir'] = os.path.join(hparams['base_path'], 'waveglow-craystack', 'data')
# Log directory for tf.summary and tensorboard
hparams['log_dir'] = os.path.join(hparams['base_path'], 'waveglow-compression', 'logs', save_name)
# Checkpoint directory to save and restore model
hparams['checkpoint_dir'] = os.path.join(hparams['base_path'], 'waveglow-compression', 'checkpoints', save_name)


# In[ ]:


# Legacy or Not implemented
# hparams['train_steps'] = 100  # Not implemented

