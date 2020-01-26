#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys
root_dir, _ = os.path.split(os.getcwd())
script_dir = os.path.join(root_dir, 'scripts')
sys.path.append(script_dir)


# In[2]:


import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import tensorflow_probability as tfp


# In[ ]:


from hparams import hparams
from custom_layers_integer import Inv1x1ConvPermute, WaveNetIntegerBlock
from custom_layers_integer import DiscreteLogisticMixParametersWaveNet, FactorOutLayer
from training_utils import log_sum_exp


# In[ ]:


class WaveGlowInteger(tf.keras.Model):
  """
  Waveglow implementation using the Invertible1x1Conv custom layer and 
  the WaveNet custom block 
  """
  
  def __init__(self, hparams, **kwargs):
    super(WaveGlowInteger, self).__init__(dtype=hparams['ftype'], **kwargs)
    
    assert(hparams['n_group'] % 2 == 0)
    self.n_flows = hparams['n_flows']
    self.n_group = hparams['n_group']
    self.n_early_every = hparams['n_early_every']
    self.n_early_size = hparams['n_early_size']
    self.upsampling_size = hparams['upsampling_size']
    self.hidden_channels = hparams['hidden_channels']
    self.hparams = hparams
    self.normalisation = hparams['train_batch_size'] * hparams['segment_length']
    self.batch_size = hparams['train_batch_size']
    self.seed = hparams['seed']
    
    # Added for IDF
    self.seeds = tf.random.uniform([self.n_flows], maxval=pow(2,16), seed=self.seed, dtype=tf.int32)
    self.n_logistic_in_mixture = hparams["n_logistic_in_mixture"]
    self.blocks = int(self.n_flows / self.n_early_every)
    self.n_factorized_channels = self.n_early_size
    self.xdim = hparams['xdim']
    self.zdim = hparams['zdim']
    self.compressing = False
    self.last_log_shift = hparams['last_log_scale']

    self.waveNetIntegerBlocks = []
    self.inv1x1ConvPermuteLayers = []
    
    # Added for IDF
    self.factorOutLayers = []
    self.discreteLogisticMixParametersNets = []
      
    n_fourth = self.n_group // 4
    n_remaining_channels = self.n_group
    block = 0
    
    for index in range(self.n_flows):
      if ((index % self.n_early_every == 0) and (index > 0)):
        n_fourth -= self.n_early_size // 4
        n_remaining_channels -= self.n_early_size
        
        self.factorOutLayers.append(
          FactorOutLayer(
            n_remaining_channels = n_remaining_channels,
            n_early_size = self.n_early_size))
        
        self.discreteLogisticMixParametersNets.append(
          DiscreteLogisticMixParametersWaveNet(
            n_factorized_channels=self.n_early_size,
            n_logistic_in_mixture=self.n_logistic_in_mixture,
            n_channels=hparams['n_channels'],
            n_layers=hparams['n_layers'],
            kernel_size=hparams['kernel_size'],
            dtype=hparams['ftype']))
        
        block += 1
        
    
      self.inv1x1ConvPermuteLayers.append(
          Inv1x1ConvPermute(
            filters=n_remaining_channels,
            seed=self.seeds[index],
            dtype=hparams['ftype'],
            name="newInv1x1conv_{}".format(index)))
      
      self.waveNetIntegerBlocks.append(
        WaveNetIntegerBlock(n_in_channels=n_fourth, 
                     n_channels=hparams['n_channels'],
                     n_layers=hparams['n_layers'],
                     kernel_size=hparams['kernel_size'],
                     dtype=hparams['ftype'],
                     name="waveNetIntegerBlock_{}".format(index)))
      
      
    self.n_remaining_channels = n_remaining_channels
    self.n_blocks = block
    
    
  def call(self, inputs, training=None):
    """
    Evaluate model against inputs
    """
    
    audio = inputs['wav']
    
    audio = layers.Reshape(
      target_shape = [self.hparams["segment_length"] // self.n_group,
                      self.n_group],
      dtype=self.dtype) (audio)
    
    output_audio = []
    output_means = []
    output_log_scales = []
    output_logit = []
    
    n_remaining_channels = self.n_group
    block = 0
    
    for index in range(self.n_flows):
      
      if ((index % self.n_early_every == 0) and (index > 0)):
        n_remaining_channels -= self.n_early_size
        
        audio, output_chunk = self.factorOutLayers[block](audio, forward=True)
        
        logit, means, log_scales = self.discreteLogisticMixParametersNets[block](audio, training=training)
        
        output_audio.append(output_chunk)
        output_logit.append(logit)
        output_means.append(means)
        output_log_scales.append(log_scales)
        block += 1
        
      audio = self.inv1x1ConvPermuteLayers[index](audio)
      audio = self.waveNetIntegerBlocks[index](audio, 
                                               forward=True,
                                               training=training)
      
    # Last factored out audio will be encoded as discrete logistic
    # The parameters are fixed and no mixture. To implement clean loss
    # easier to generate mix of the same discrete logistic
    audio = tf.reshape(audio, [audio.shape[0], audio.shape[1] * audio.shape[2], 1])
    
    last_means = tf.zeros(audio.shape[:-1] + [self.n_logistic_in_mixture])
    last_log_scales = tf.zeros(audio.shape[:-1] + [self.n_logistic_in_mixture]) + self.last_log_shift
    last_logit = tf.concat([tf.ones(audio.shape), 
                            tf.zeros(audio.shape[:-1] + [self.n_logistic_in_mixture - 1])],
                           axis=2)
    
    # Append last outputs
    output_audio.append(audio)
    output_logit.append(last_logit)
    output_means.append(last_means)
    output_log_scales.append(last_log_scales)
    
    # Concatenate outputs
    output_means = tf.concat(output_means, axis=1)
    output_logit = tf.concat(output_logit, axis=1)
    output_log_scales = tf.concat(output_log_scales, axis=1)
    output_audio = tf.concat(output_audio, axis=1)
    
    return (output_audio, output_logit, output_means, output_log_scales)
    
  def generate(self, signal, block):
    
    n_factorized_channels = ((self.n_group - self.n_remaining_channels) // self.n_early_every)
    
    target_shape = [self.batch_size,
                  (self.hparams["segment_length"] // self.n_group),
                  self.n_remaining_channels + (self.n_blocks - block) * n_factorized_channels]
    
    audio = tf.reshape(signal, target_shape)
    
    for index in list(reversed(range(self.n_flows)))[(self.n_blocks - block) * self.n_early_every:(self.n_blocks - block + 1) * self.n_early_every]:
      audio = self.waveNetIntegerBlocks[index](audio, forward=False)
      audio = self.inv1x1ConvPermuteLayers[index](audio, forward=False)
      
    if block - 1 >= 0:
      logits, means, log_scales = self.discreteLogisticMixParametersNets[block-1](audio)
    else:
      means = tf.zeros([self.batch_size, self.hparams["segment_length"] // self.n_remaining_channels, self.n_logistic_in_mixture])
      log_scales = tf.zeros_like(means) + self.last_log_shift
      logits = tf.concat([tf.ones([self.batch_size, self.hparams["segment_length"] // self.n_remaining_channels, 1]), 
                          tf.zeros([self.batch_size, self.hparams["segment_length"] // self.n_remaining_channels, self.n_logistic_in_mixture - 1])],
                          axis=2)  
    
    target_shape = [self.batch_size,-1,1]
    
    audio = tf.reshape(audio, target_shape)
    
    means_shape = [self.batch_size, 
                   (self.hparams["segment_length"] // self.n_group) * n_factorized_channels,
                   1]
    
    return audio, (logits, means, log_scales)
  
  def infer_craystack(self, inputs, training=None):
    
    audio, logits, means, log_scales = self.call(inputs=inputs, training=training)
    
    audio = [(tf.squeeze(x, axis=-1).numpy() * pow(2, 15)) + pow(2, 15) for x in tf.split(audio, self.n_blocks + 1, axis=1)]
    all_params = [(tf.transpose(x, [0, 2, 1]).numpy().astype(np.float32), 
                  tf.transpose(y, [0, 2, 1]).numpy().astype(np.float32), 
                  tf.transpose(z, [0, 2, 1]).numpy().astype(np.float32)) for x,y,z in zip(tf.split(logits, self.n_blocks + 1, axis=1),
                                                                  tf.split(means, self.n_blocks + 1, axis=1),
                                                                  tf.split(log_scales, self.n_blocks + 1, axis=1))]
    
    return audio, all_params
  
  def generate_craystack(self, x=None, z=None, block=4):
    
    if block == 4:
      means = tf.zeros([self.batch_size, np.prod(self.zdim), self.hparams["n_logistic_in_mixture"]])
      log_scales = tf.zeros_like(means) + self.last_log_shift
      logits = tf.concat([tf.ones([self.batch_size, np.prod(self.zdim), 1]),
                          tf.zeros([self.batch_size, np.prod(self.zdim), self.hparams["n_logistic_in_mixture"] - 1])], axis=2)
    else:
      x = None if x is None else x
      z = tf.convert_to_tensor((z - pow(2, 15)) / pow(2, 15) , dtype=tf.float32)
      signal = z if x is None else tf.concat([tf.reshape(z, shape=[self.batch_size, 
                                               self.hparams['segment_length'] // self.n_group, 
                                               self.n_remaining_channels]),
                                              tf.reshape(x, shape=[self.batch_size, 
                                              self.hparams['segment_length'] // self.n_group, 
                                              (self.n_blocks - block) * self.n_remaining_channels])],
                                             axis=2)
      x, (logits, means, log_scales) = self.generate(signal=signal, block=block)
    
    logits = tf.transpose(logits, [0,2,1])
    means = tf.transpose(means, [0,2,1])
    log_scales = tf.transpose(log_scales, [0,2,1])
      
    return x, (logits.numpy().astype(np.float32), means.numpy().astype(np.float32), log_scales.numpy().astype(np.float32))
  
  def select_means_and_scales(self, logits, means, log_scales, output_shape):

    argmax = tf.argmax(tf.nn.softmax(logits), axis=2)
    selector = tf.one_hot(argmax, 
                          depth=logits.shape[2],
                          dtype=tf.float32)

    scales = tf.reduce_sum(tf.math.exp(log_scales) * selector, axis=2)
    means = tf.reduce_sum(means * selector, axis=2)

    return tf.reshape(means, output_shape), tf.reshape(scales, output_shape)
  
  def set_compression(self):
    self.compressing = True
    self.batch_size = self.hparams['compress_batch_size']
    
  def set_training(self):
    self.compressing = False
    self.batch_size = self.hparams['train_batch_size']
  
  def get_config(self):
    config = super(WaveGlow, self).get_config()
    config.update(hparams = hparams)
    
    return config
  
  def sample_from_discretized_mix_logistic(self, logits, means, log_scales, log_scale_min=-8.):
    '''
    Args:
        logits, means, log_scales: Tensor, [batch_size, time_length, n_logistic_in_mixture]
    Returns:
        Tensor: sample in range of [-1, 1]
    
    Adapted from pixelcnn++
    '''
    # sample mixture indicator from softmax
    temp = tf.random.uniform(logits.shape, minval=1e-5, maxval=1. - 1e-5)
    temp = logits - tf.math.log(-tf.math.log(temp))
    argmax = tf.math.argmax(temp, -1)

    # [batch_size, time_length] -> [batch_size, time_length, nr_mix]
    one_hot = tf.one_hot(argmax, 
                         depth=self.n_logistic_in_mixture,
                         dtype=tf.float32)

    # select logistic parameters
    means = tf.reduce_sum(means * one_hot, axis=-1)
    log_scales = tf.maximum(tf.reduce_sum(
      log_scales * one_hot, axis=-1), log_scale_min)

    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8-bit value when sampling
    u = tf.random.uniform(means.shape, minval=1e-5, maxval=1. - 1e-5) 
    x = means + tf.math.exp(log_scales) * (tf.math.log(u) - tf.math.log(1 - u))
    
    return tf.minimum(tf.maximum(x, -1. + 1e-5), 1. - 1e-5)
  
  def total_loss(self, outputs, num_classes=65536, log_scale_min=-7.0):
    '''
    Discretized mix of logistic distributions loss.
    Note that it is assumed that input is scaled to [-1, 1]
    Adapted from pixelcnn++
    '''

    # audio[batch_size, time_length, 1] + unpack parameters: [batch_size, time_length, num_mixtures]
    output_audio, logit_probs, means, log_scales = outputs
    
    # output_audio [batch_size, time_length, 1] -> [batch_size, time_length, num_mixtures]
    y = output_audio * tf.ones(shape=[1, 1, self.n_logistic_in_mixture], dtype=self.dtype)

    centered_y = y - means
    inv_stdv = tf.math.exp(-log_scales)
    plus_in = inv_stdv * (centered_y + 1. / (num_classes - 1))
    cdf_plus = tf.nn.sigmoid(plus_in)
    min_in = inv_stdv * (centered_y - 1. / (num_classes - 1))
    cdf_min = tf.nn.sigmoid(min_in)

    log_cdf_plus = plus_in - tf.nn.softplus(plus_in) # log probability for edge case of -32768 (before scaling)
    log_one_minus_cdf_min = - tf.nn.softplus(min_in) # log probability for edge case of 32767 (before scaling)

    #probability for all other cases
    cdf_delta = cdf_plus - cdf_min

    mid_in = inv_stdv * centered_y
    #log probability in the center of the bin, to be used in extreme cases
    log_pdf_mid = mid_in - log_scales - 2. * tf.nn.softplus(mid_in)
    
    log_probs = tf.where(y < -0.999, log_cdf_plus,
                         tf.where(y > 0.999, log_one_minus_cdf_min,
                                  tf.where(cdf_delta > 1e-5,
                                           tf.math.log(tf.maximum(cdf_delta, 1e-12)),
                                           log_pdf_mid - np.log((num_classes - 1) / 2))))

    log_probs = log_probs + tf.nn.log_softmax(logit_probs, axis=-1)
    
    logistic_loss = -tf.reduce_sum(log_sum_exp(log_probs)) / (y.shape[0] * y.shape[1] * y.shape[2])
    
    tf.summary.scalar(name='total_loss',
                      data=(logistic_loss))
    
    for block, log_scales in enumerate(tf.split(log_scales, 4, axis=1)[:-1]):
      tf.summary.scalar(name=f'mean_log_scales_block{block}',
                        data=tf.reduce_mean(log_scales),
                        step=tf.summary.experimental.get_step())
      tf.summary.scalar(name=f'max_log_scales_block{block}',
                        data=tf.reduce_max(log_scales),
                        step=tf.summary.experimental.get_step())
      tf.summary.scalar(name=f'min_log_scales_block{block}',
                        data=tf.reduce_min(log_scales),
                        step=tf.summary.experimental.get_step())
    
    return logistic_loss


# In[ ]:





# In[ ]:




