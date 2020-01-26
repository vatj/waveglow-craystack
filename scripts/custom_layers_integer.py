#!/usr/bin/env python
# coding: utf-8

# # Invertible Convolution and WaveNet Custom Layers 

# ## Boilerplate
# Start with standard imports as well as adding the scripts directory to the system path to allow custom imports.

# In[1]:


import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


# In[2]:


import os, sys
root_dir, _ = os.path.split(os.getcwd())
script_dir = os.path.join(root_dir, 'scripts')
sys.path.append(script_dir)


# In[3]:


from hparams import hparams
from training_utils import round_ste


# ## Invertible Convolution where the kernel is a permutation matrix and is not trainable.
# 
# The forward boolean in the call method can be used to run the layer in reverse.

# In[ ]:


class Inv1x1ConvPermute(layers.Layer):
  """
  The kernel of this convolution is just a permutation
  matrix. The permutation is initialised and kept fixed
  during training and evaluation. Contrary to continuous
  waveglow there is no added loss.
  """
  
  def __init__(self, filters, seed, **kwargs):
    super(Inv1x1ConvPermute, self).__init__(**kwargs)
    self.filters = filters
    self.seed = seed
    
    
  def build(self, input_shape):
    
    self.kernel = self.add_weight(
      shape=[self.filters, self.filters],
      initializer="identity", 
      trainable=False,
      name="kernel")
    
    # Random permutation. Some combinations lead to a more or less stable training process
    self.kernel = tf.expand_dims(tf.random.shuffle(self.kernel, seed=self.seed), axis=0)
    
    self.inverse_kernel = tf.cast(tf.linalg.inv(
          tf.cast(self.kernel, tf.float64)), dtype=self.dtype)
    
    self.built = True
  
  def call(self, inputs, forward=True):
    if forward:
      return tf.nn.conv1d(inputs, self.kernel, 
                          stride=1, padding='SAME')
      
    else:
      return tf.nn.conv1d(inputs, self.inverse_kernel, 
                          stride=1, padding='SAME')
    
  def get_config(self):
    config = super(Inv1x1ConvPermute, self).get_config()
    config.update(filters = self.filters)
    config.update(seed = self.seed)
    
    return config


# ## Nvidia WaveNet Implementation
# Difference with the original implementations :
# WaveNet convonlution need not be causal. 
# No dilation size reset. 
# Dilation doubles on each layer

# In[ ]:


class WaveNetNvidia(layers.Layer):
  """
  Wavenet Block as defined in the WaveGlow implementation from Nvidia
  
  WaveNet convonlution need not be causal. 
  No dilation size reset. 
  Dilation doubles on each layer.
  """
  def __init__(self, n_in_channels, n_channels = 256, 
               n_layers = 12, kernel_size = 3, **kwargs):
    super(WaveNetNvidia, self).__init__(**kwargs)
    
    assert(kernel_size % 2 == 1)
    assert(n_channels % 2 == 0)
    
    self.n_layers = n_layers
    self.n_channels = n_channels
    self.n_in_channels = n_in_channels
    self.kernel_size = kernel_size
    
    self.in_layers = []
    self.res_skip_layers = []
    
    self.start = layers.Conv1D(filters=self.n_channels,
                               kernel_size=1,
                               dtype=self.dtype,
                               name="start")
    
    self.end = layers.Conv1D(
      filters=3 * self.n_in_channels,
      kernel_size = 1,
      kernel_initializer=tf.initializers.zeros(),
      bias_initializer=tf.initializers.zeros(),
      activation=tf.nn.tanh,
      dtype=self.dtype,
      name="end")

    for index in range(self.n_layers):
      dilation_rate = 2 ** index
      in_layer = layers.Conv1D(filters=2 * self.n_channels,
                    kernel_size= self.kernel_size,
                    dilation_rate=dilation_rate,
                    padding="SAME",
                    dtype=self.dtype,
                    name="conv1D_{}".format(index))
     
      self.in_layers.append(in_layer)
      
      if index < self.n_layers - 1:
        res_skip_channels = 2 * self.n_channels
      else:
        res_skip_channels = self.n_channels
        
      res_skip_layer = layers.Conv1D(
        filters=res_skip_channels,
        kernel_size=1,
        dtype=self.dtype,
        name="res_skip_{}".format(index))
      
      self.res_skip_layers.append(res_skip_layer)
      
    
  def call(self, inputs, training=False):
    """
    This implementatation does not require exposing a training boolean flag 
    as only the integer coupling behaviour needs reversing during
    inference.
    """
    audio_0 = inputs
    
    started = self.start(audio_0)
    
    
    for index in range(self.n_layers):
      in_layered = self.in_layers[index](started)
      
      half_tanh, half_sigmoid = tf.split(
        in_layered, 2, axis=2)
      half_tanh = tf.nn.tanh(half_tanh)
      half_sigmoid = tf.nn.sigmoid(half_sigmoid)
    
      activated = half_tanh * half_sigmoid
      
      res_skip_activation = self.res_skip_layers[index](activated)
      
      if index < (self.n_layers - 1):
        res_skip_activation_0, res_skip_activation_1 = tf.split(
          res_skip_activation, 2, axis=2)
        started = res_skip_activation_0 + started
        skip_activation = res_skip_activation_1
      else:
        skip_activation = res_skip_activation

      if index == 0:
        output = skip_activation
      else:
        output = skip_activation + output
        
    output = self.end(output)
    
    # Added rounding operation with straight through gradient (ste)
    output = round_ste(output * pow(2, 15)) / pow(2, 15)
    
    return output
  
  def get_config(self):
    config = super(WaveNetBlock, self).get_config()
    config.update(n_in_channels = self.n_in_channels)
    config.update(n_channels = self.n_channels)
    config.update(n_layers = self.n_layers)
    config.update(kernel_size = self.kernel_size)
  
    return config


# ## Integer Translation Coupling Layer
# 
# In the IDF paper, the affine coupling layer is replaced by a additive layer which differ in a couple of ways:
# - Rounding operation applied to the output of the wavenet neural network
# - No multiplication

# In[ ]:


class IntegerCoupling(layers.Layer):
  """
  Invertible Integer Coupling Layer.
  Since inputs are between -1,1 we 
  The inverted behaviour is obtained by setting the forward boolean
  in the call method to false.
  """
  
  def __init__(self, **kwargs):
    super(IntegerCoupling, self).__init__(**kwargs)
    
  def call(self, inputs, forward=True):
    
    audio_1, wavenet_output = inputs
    
    if forward:
      audio_1 = (audio_1 + wavenet_output) % 2 - tf.ones(audio_1.shape, dtype=audio_1.dtype)
    else:
      audio_1 = (audio_1 - wavenet_output) % 2 - tf.ones(audio_1.shape, dtype=audio_1.dtype)     
        
    return audio_1
  
  def get_config(self):
    config = super(IntegerCoupling, self).get_config()
    
    return config


# ## WaveNet And Integer Coupling
# This block is a convenience block which has been defined to make it more straightforward to implement the WaveGlow model using the keras functional API. This version uses: 
# - Integer coupling instead of the original affine coupling
# - The conditioning is applied to 25% of the inputs instead of the original 50%

# In[ ]:


class WaveNetIntegerBlock(layers.Layer):
  """
  Wavenet + Integer Coupling Layer
  Convenience block to provide a tidy model definition
  """
  def __init__(self, n_in_channels, n_channels = 256,
               n_layers = 12, kernel_size = 3, **kwargs):
    super(WaveNetIntegerBlock, self).__init__(**kwargs)
    
    self.n_layers =  n_layers
    self.n_channels = n_channels
    self.n_in_channels = n_in_channels
    self.kernel_size = kernel_size
    
    self.wavenet = WaveNetNvidia(n_in_channels=n_in_channels,
                                 n_channels=n_channels,
                                 n_layers=n_layers,
                                 kernel_size=kernel_size,
                                 dtype=self.dtype)
    
    self.integer_coupling = IntegerCoupling(dtype=self.dtype)
      
    
  def call(self, inputs, forward=True, training=False):
    """
    forward should be set to false to inverse integer layer
    Split audio 75%-25% to make it 
    """
    
    splits = [self.n_in_channels,
              3 * self.n_in_channels]
    
    audio_0, audio_1 = tf.split(inputs, splits, axis=2)
    
    wavenet_output = self.wavenet(audio_0, training=training)
    
    audio_1 = self.integer_coupling(
      (audio_1, wavenet_output), forward=forward)   
         
    audio = layers.Concatenate(
      axis=2, 
      dtype=self.integer_coupling.dtype) ([audio_0, audio_1])
    
    return audio
  
  def get_config(self):
    config = super(WaveNetIntegerBlock, self).get_config()
    config.update(n_in_channels = self.n_in_channels)
    config.update(n_channels = self.n_channels)
    config.update(n_layers = self.n_layers)
    config.update(kernel_size = self.kernel_size)
  
    return config


# In[ ]:


class FactorOutLayer(layers.Layer):
  """
  Factor Out layer implementation.
  TODO :
  - Implement training boolean
  """
  
  def __init__(self, n_remaining_channels, n_early_size,
               **kwargs):
    super(FactorOutLayer, self).__init__(**kwargs)
    
    self.n_remaining_channels = n_remaining_channels
    self.n_early_size = n_early_size
    
    
    
  def call(self, audio, forward=True):
    
    if forward is True:     
      audio = layers.Permute(dims=(2, 1), dtype=self.dtype) (audio)

      output_chunk = layers.Cropping1D(
        cropping=(0, self.n_remaining_channels),
        dtype=self.dtype) (audio)

      audio = layers.Cropping1D(
        cropping=(self.n_early_size, 0),
        dtype=self.dtype) (audio)

      audio = layers.Permute(dims=(2, 1), dtype=self.dtype) (audio)

      output_chunk = layers.Permute(dims=(2, 1), 
                                    dtype=self.dtype) (output_chunk)
      
      output_chunk = tf.reshape(output_chunk, 
                                [output_chunk.shape[0], 
                                output_chunk.shape[1] * output_chunk.shape[2], 
                                1])
      
      return audio, output_chunk
    else:
      raise NotImplementedError('The false forward boolean for this layer is not working yet')
  
  def get_config(self):
    config = super(FactorOutLayer, self).get_config()
    config.update(n_remaining_channels = self.n_remaining_chanels)
    config.update(n_early_size = self.n_early_size)
    
    return config


# In[ ]:


class DiscreteLogisticMixParametersWaveNet(layers.Layer):
  """
  Wavenet Block based on Nvidia implementation. 
  Modified to output logistic mixture parameters
  No rounding operation
  
  WaveNet convonlution need not be causal. 
  No dilation size reset. 
  Dilation doubles on each layer.
  """
  def __init__(self, n_factorized_channels, n_logistic_in_mixture,
               n_channels = 256, n_layers = 12, kernel_size = 3, **kwargs):
    super(DiscreteLogisticMixParametersWaveNet, self).__init__(**kwargs)
    
    assert(kernel_size % 2 == 1)
    assert(n_channels % 2 == 0)
    
    self.n_layers = n_layers
    self.n_logistic_in_mixture = n_logistic_in_mixture
    self.n_channels = n_channels
    self.n_factorized_channels = n_factorized_channels
    self.kernel_size = kernel_size
    
    self.in_layers = []
    self.res_skip_layers = []
    
    self.start = layers.Conv1D(filters=self.n_channels,
                               kernel_size=1,
                               dtype=self.dtype,
                               name="start")
    
    self.end = layers.Conv1D(
      filters=3*n_factorized_channels*n_logistic_in_mixture,
      kernel_size = 1,
      kernel_initializer=tf.initializers.GlorotUniform(),
      bias_initializer=tf.initializers.zeros(),
      dtype=self.dtype,
      name="end")

    for index in range(self.n_layers):
      dilation_rate = 2 ** index
      in_layer = layers.Conv1D(filters=2 * self.n_channels,
                    kernel_size= self.kernel_size,
                    dilation_rate=dilation_rate,
                    padding="SAME",
                    dtype=self.dtype,
                    name="conv1D_{}".format(index))
     
      self.in_layers.append(in_layer)
      
      if index < self.n_layers - 1:
        res_skip_channels = 2 * self.n_channels
      else:
        res_skip_channels = self.n_channels
        
      res_skip_layer = layers.Conv1D(
        filters=res_skip_channels,
        kernel_size=1,
        dtype=self.dtype,
        name="res_skip_{}".format(index))
      
      self.res_skip_layers.append(res_skip_layer)
      
    
  def call(self, inputs, training=False):
    
    started = self.start (inputs)
    
    for index in range(self.n_layers):
      in_layered = self.in_layers[index](started)
      
      half_tanh, half_sigmoid = tf.split(
        in_layered, 2, axis=2)
      half_tanh = tf.nn.tanh(half_tanh)
      half_sigmoid = tf.nn.sigmoid(half_sigmoid)
    
      activated = half_tanh * half_sigmoid
      
      res_skip_activation = self.res_skip_layers[index](activated)
      
      if index < (self.n_layers - 1):
        res_skip_activation_0, res_skip_activation_1 = tf.split(
          res_skip_activation, 2, axis=2)
        started = res_skip_activation_0 + started
        skip_activation = res_skip_activation_1
      else:
        skip_activation = res_skip_activation

      if index == 0:
        output = skip_activation
      else:
        output = skip_activation + output
        
    output = self.end (output)
    
    logits, means, log_scales = tf.split(output, 3, axis=2)
    
    target_shape = [logits.shape[0], logits.shape[1] * self.n_factorized_channels, self.n_logistic_in_mixture]
    
    logits = tf.reshape(logits, target_shape)
    means = tf.reshape(means, target_shape)
    log_scales = tf.reshape(log_scales, target_shape)
    
    return logits, means, log_scales
  
  def get_config(self):
    config = super(WaveNetBlock, self).get_config()
    config.update(n_factorized_channels = self.n_factorized_channels)
    config.update(n_logistic_in_mixture = self.n_logistic_in_mixture)
    config.update(n_channels = self.n_channels)
    config.update(n_layers = self.n_layers)
    config.update(kernel_size = self.kernel_size)
  
    return config

