# Integer Discrete Flow for Audio Compression in Tensorflow2
[License Badge](https://img.shields.io/github/license/vatj/waveglow-craystack?style=plastic)
Implementation of a Integer Discrete Flow to perform lossless audio compression using Tensorflow 2.0. The flow model is adapted from PyTorch implementations of [Nvidia WaveGlow model by Prender et al.](https://arxiv.org/abs/1811.00002) and [Integer Discrete Flow by Hoogeboom et al.](https://arxiv.org/abs/1905.07376). The neural net takes the audio signal as input and map it on a pair (signal, parameters) whose parameters encode a hierarchical mixture of logistic distributions. Given the parameters, the mapped signal can be encoded efficiently using entropy coding, here range Asymmetric Numeral Systems. The encoder implementation is from [Craystack](https://openreview.net/forum?id=r1lZgyBYwS).

Note that because of the hierarchical structure, an approximation of the full audio file can be obtained with increasing precision for each block of the data recovered. Examples coming soon.


Results :
Original : WAV file, audio signal encoded using 16 bits integer.
FLAC : Compression ratio _ , bits per dimension _
Small version (2^8 points signal): Compression ratio 2.02 , bits per dimension 7.92
Medium version (2^10 points signal) : Compression ratio 1.62 , bits per dimension 9.88  (in training)
Large version (2^14 points signal): Compression ratio 1.69 , bits per dimension 9.47 (in training)

Disclaimer : Encoding/Decoding time and memory requirement is currently unpractical. No hyperparameter fine-tuning has been attempted. Training has been performed using the [ljspeech dataset](https://keithito.com/LJ-Speech-Dataset/) which contains speech audio samples, not music.

## Loss, compression ratio and bit_per_dimension plots for the small model


### Compression Ratio
![Compression ration](/assets/compression_ratio.svg)

### Bits per dimension
![Bits per dimension](/assets/bits_per_dim.svg)

### Total Loss
![Loss Plot](/assets/total_loss.svg)

Note that the discontinuity is due to increasing the hyperparameter associated with sharpness of the logistic distribution in the last layer during training and not to instability. Instability were observed for similar hyperparameters but different random seed suggesting the permutation in the convolutional layers play an important role in the stability of the model.

## Quickstart

```shell
git clone --recurse-submodules git@github.com:vatj/waveglow-craystack.git
cd waveglow-craystack
mkdir data logs checkpoints
```

Fill in machine specific parameters in the hparams.py script. The hparams dict is loaded from the script, not the corresponding notebook.

Run data preprocessing script raw_ljspeech_to_tfrecords. Modify the training script to specify the GPU (os.environ variable). Run the training script. 

```shell
python scripts/raw_ljspeech_to_tfrecords.py
python scripts/training_main.py
```

Use tensorboard to monitor training. 


## Content

- All python code exist in the notebooks directory.
- The scripts directory contains a python script version of all notebooks.
- The submodules directory contains the craystack implementation of the entropy coder
- The assets directory contains a few images from tensorboard

Specific part of the code :
- The neural net model can be found in waveglow_model_integer
- The custom layers can be found in custom_layers_integer
- The main training notebook is training_integer
- The interface with craystack can be found in waveglow_codec
- The data pipeline is found in raw_ljspeech_to_tfrecords
- All parameters are taken from the hparams.py (NOT from the notebook)
- Trainig_utils contains useful code snippets


## TODOS

- [ ] Add signal reconstruction examples
- [ ] Add results flac
