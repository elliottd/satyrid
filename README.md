# satyrid

An attention-based image description model.

This code is based on Kelvin Xu's [arctic
captions](https://github.com/kelvinxu/arctic-captions) described in
[Show, Attend and Tell: Neural Image Caption Generation with Visual
Attention](http://arxiv.org/abs/1502.03044).

Changes:

* Create input data from a directory of images and a JSON
  file concerning the descriptions.
* Gradient norm and value clipping.
* Most recent version of the ADAM optimiser (v8).
* Monitor training performance using external metrics.

## Dependencies

* Python 2.7
* [Theano](http://www.deeplearning.net/software/theano)
* [NumPy](http://www.numpy.org/)
* [scikit learn](http://scikit-learn.org/stable/index.html)
* [skimage](http://scikit-image.org/docs/dev/api/skimage.html)
* [PyTables](http://www.pytables.org/) (for reading the image features)

To extract visual features from your own images and to create
training, validation, and test inputs files, you will need:

* [Caffe](http://www.caffe.org/) built with the Python bindings (if
  you want to extract visual features by yourself)

To use the evaluation script (`metrics.py`): see
[coco-caption](https://github.com/tylin/coco-caption) for the
requirements.  Install `coco-caption` in `evaluate/` and create an
empty `__init__.py` in `evaluate/` so it can be imported as a module.

## Data & Pre-trained Model

You can download pre-extracted [training](https://staff.fnwi.uva.nl/d.elliott/satyrid/train-cnn_features.hdf5), [dev](https://staff.fnwi.uva.nl/d.elliott/satyrid/dev-cnn_features.hdf5), and [test](https://staff.fnwi.uva.nl/d.elliott/satyrid/test-cnn_features.hdf5) features for the Flickr30K dataset:

* The HDF5 file contains the CONV_5,4 image feature vectors. Each
  image vector is stored as a flattened (14, 14, 512) `ndarray`, which
  will be reshaped into a (14x14, 512) `ndarray` when it is used by
  the model.

You can download the pre-extracted [training](https://staff.fnwi.uva.nl/d.elliott/satyrid/train.pkl), [development](https://staff.fnwi.uva.nl/d.elliott/satyrid/dev.pkl), and [test](https://staff.fnwi.uva.nl/d.elliott/satyrid/test.pkl) sentences and [the dictionary](https://staff.fnwi.uva.nl/d.elliott/satyrid/dictionary.pkl).

* The numpy file contains a list of ((sentence, index)) tuples. The
  `index` entry is directly mapped to the `index` of the visual
  feature vector in the HDF5 file.

You can also [download a pre-trained model](https://staff.fnwi.uva.nl/d.elliott/satyrid/flickr30k.model.tgz) for this dataset.
  
## Creating new dataset objects

`make_dataset.py` takes care of creating the image features file and
the sentences file.  See `make_dataset.py` for instructions on how to
create dataset files from your data.

If you create a new dataset, you will need to create a new dataset
loader module to work with your new dataset. See `flickr30k.py` for
how to do this.

## Training a model

You can train a model using `THEANO_FLAGS=floatX=float32 python
train_model.py`.  See the documentation in `train_model.py` and
`model.py` for more information on the options.

If you want to use the `metrics.py` script to control training the
model (e.g. save model parameters based on Meteor or CIDEr), then pass
`"{'use_metrics':'True'}"` as an argument to `train_model.py` and
install the dependencies for the
[coco-caption](https://github.com/tylin/coco-caption) for the
requirements.

## Generating descriptions

Generate descriptions using `THEANO_FLAGS=floatX=float32 python
generate_caps.py $model_name $PREFIX`. This will generate descriptions
into `$PREFIX.dev.txt` and `$PREFIX.test.txt`. Use the `--dataset
$DATASET_NAME` argument to generate descriptions of images in a
different dataset.

## Reference

If you use this code as part of any published research, please
acknowledge the following paper (it encourages researchers who publish
their code!):

**"Show, Attend and Tell: Neural Image Caption Generation with Visual
Attention."** Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron
Courville, Ruslan Salakhutdinov, Richard Zemel, Yoshua Bengio. ICML
(2015)

## License

The code is released under a [revised (3-clause) BSD
License](http://directory.fsf.org/wiki/License:BSD_3Clause).
