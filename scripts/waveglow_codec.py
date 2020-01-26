#!/usr/bin/env python
# coding: utf-8

# # Sanity Check Notebook (should be converted to test unit)

# ## Boilerplate

# In[ ]:


import os, sys
root_dir, _ = os.path.split(os.getcwd())
script_dir = os.path.join(root_dir, 'scripts')
sys.path.append(script_dir)
os.nice(15)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# In[ ]:


import tensorflow as tf
import numpy as np
import time


# In[ ]:


import craystack as cs
from craystack.codecs import LogisticMixture_UnifBins, Codec
import training_utils as utils


# In[ ]:


def Waveglow_codec(model, hparams):
  
  coding_prec = hparams['coding_prec']
  bin_precision = hparams['bin_precision']
  bin_lowerbound = hparams['bin_lower_bound']
  bin_upperbound = hparams['bin_upper_bound']
  
  def WaveglowLogisticMixture(all_params, block):
    return LogisticMixture_UnifBins(logit_probs=all_params[0], means=all_params[1], log_scales=all_params[2], 
                                    coding_prec=coding_prec, bin_prec=bin_precision, 
                                    bin_lb=bin_lowerbound, bin_ub=bin_upperbound)
  
  
  def AutoRegressiveIDF(model, elem_codec):
    """
    Codec for data from distributions which are calculated autoregressively.
    That is, the data can be partitioned into n elements such that the
    distribution/codec for an element is only known when all previous
    elements are known. This is does not affect the push step, but does
    affect the pop step, which must be done in sequence (so is slower).
    elem_param_fn maps data to the params for the respective codecs.
    elem_idxs defines the ordering over elements within data.
    We assume that the indices within elem_idxs can also be used to index
    the params from elem_param_fn. These indexed params are then used in
    the elem_codec to actually code each element.
    """
    def push(message, data):
      encodable, all_params = model.infer_craystack(data)
      for block in range(model.n_blocks + 1):
        elem_params = all_params[block]
        elem_push, _ = elem_codec(elem_params, block) # block potentially useless here but good to have the option
        message = elem_push(message, encodable[block].astype('uint64'))
      return message

    def pop(message):
      elem = None
      for block in reversed(range(model.n_blocks + 1)):
        data, all_params = model.generate_craystack(x=None if block + 1 > model.n_blocks else data, 
                                                    z=elem, block=block+1)
        _, elem_pop = elem_codec(all_params=all_params, block=block)
        message, elem = elem_pop(message)
        
      data, all_params = model.generate_craystack(x=data, z=elem, block=0)
      return message, data
  
    return Codec(push, pop)

  return AutoRegressiveIDF(model=model, elem_codec=WaveglowLogisticMixture)


# In[ ]:


def compress_samples(model, hparams, step=tf.constant(0), decode=False):

  model.set_compression()
  
  test_set = utils.load_training_files_tfrecords(
    record_pattern=os.path.join(hparams['tfrecords_dir'], hparams['train_files'] + '*'))
      
  datapoints = list(test_set.unbatch().batch(
    hparams['compress_batch_size']).take(hparams['n_compress_datapoint']))
  
  num_pixels = hparams['n_compress_datapoint'] * hparams['compress_batch_size'] * hparams['segment_length']
  
  ## Load Codec
  waveglow_append, waveglow_pop = cs.repeat(Waveglow_codec(model=model, hparams=hparams), 
                                            hparams['n_compress_datapoint'])
  
  ## Encode
  encode_t0 = time.time()
  init_message = cs.empty_message(shape=(hparams['compress_batch_size'],
                                         hparams['segment_length'] // 4))
  
  # Encode the audio samples
  message = waveglow_append(init_message, datapoints)

  flat_message = cs.flatten(message)
  encode_t = time.time() - encode_t0

  tf.print("All encoded in {:.2f}s.".format(encode_t))

  original_len = 16 * hparams['n_compress_datapoint'] * hparams['segment_length']
  message_len = 32 * len(flat_message)
  tf.print("Used {} bits.".format(message_len))
  tf.print("This is {:.2f} bits per pixel.".format(message_len / num_pixels))
  tf.print("Compression ratio : {:.2f}".format(original_len / message_len))
  
  tf.summary.scalar(name='bits_per_dim', data= message_len / num_pixels, step=step)
  tf.summary.scalar(name='compression_ratio', data=original_len / message_len, step=step)
  
  if decode:
    ## Decode
    decode_t0 = time.time()
    message = cs.unflatten(flat_message, shape=(hparams['compress_batch_size'], 
                                                hparams['segment_length'] // 4))

    message, datapoints_ = waveglow_pop(message)
    decode_t = time.time() - decode_t0

    print('All decoded in {:.2f}s.'.format(decode_t))
    
    datacompare = [data['wav'].numpy()[..., np.newaxis] for data in datapoints]
    np.testing.assert_equal(datacompare, datapoints_)
    np.testing.assert_equal(message, init_message)
  
  model.set_training()

