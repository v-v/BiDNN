BiDNN
=====
Bidirectional (Symmetrical) Deep Neural Networks

As originally described in: V. Vukotić, C. Raymond, and G. Gravier. **Bidirectional Joint Representation Learning with Symmetrical Deep Neural Networks for Multimodal and Crossmodal Applications**. In *Proceedings of the 2016 ACM International Conference on Multimedia Retrieval (ICMR)*, pages 343–346. ACM, New York, 2016. (Available [here](https://hal.inria.fr/hal-01314302/document))

and evaluated in more details in: V. Vukotić, C. Raymond, and G. Gravier.  **Multimodal and Crossmodal Representation Learning from Textual and Visual Features with Bidirectional Deep Neural Networks for Video Hyperlinking**. In *ACM Multimedia 2016 Workshop: Vision and Language Integration Meets Multimedia Fusion (iV&L-MM'16)*, ACM, Amsterdam, 2016. (Available [here](https://hal.inria.fr/hal-01374727/document))


BiDNNs are similar to (or a variation of) multimodal autoencoders. They offer crossmodal translation and superior multimodal embedding created in a common representation created by the two crossmodal translation. The following picture illustrates the architectural differences of classical multimodal autoencoders and BiDNNs:

![Alt text](images/AEvsBiDNN.png?raw=true "Title")

Multimodal Autoencoders
-----------------------

Classical multimodal autoencoders usually come in two varieties: i) standard autoencoders that are used in a multimodal setup by contactenating the modalities and ii) multimodal autoencoders that have separate layers for each modality and one or more central fully connected layers that yield a multimodal representation and serve as a translation layer (as illustrated in the left (*a*) part of the previous figure). One modality is often sporadically removed (zeroed) while a reconstruction of both modalities is expected at its output to force the multimodal autoencoder to represent one modality from the other.

Autoencoders however have some downsides which slightly deteriorate performance:
* Both modalities influence the same central layer(s), either directly or indirectly, through other modality-specific fully connected layers. Even when translating from one modality to the other, the input modality is either mixed with the other or with a zeroed input.
* Autoencoders need to learn to reconstruct the same output both when one modality is marked missing (e.g., zeroed) and when both modalities are presented as input.
* Classical autoencoders are primarily made for multimodal embedding while crossmodal translation is offered as a secondary function.

Bidirectional (Symmetrical) Deep Neural Networks
------------------------------------------------

In bidirectional deep neural networks, learning is performed in both directions: one modality is presented as an input and the other as the expected output while at the same time the second one is presented as input and the first one as expected output. This is equivalent to using two separate deep neural networks and tying them (sharing specific weight variables) to make them symmetrical, as illustrated in the right part of the previous figure (*b*). Implementation-wise the variables representing the weights are shared across the two networks and are in fact the same variables. Learning of the two crossmodal mappings is then performed simultaneously and they are forced to be as close as possible to each other's inverses by the symmetric architecture in the middle. A joint representation in the middle of the two crossmodal mappings is also formed while learning.

Given such an architecture, crossmodal translation is done straightforwardly by presenting the first modality and obtaining the output in the representation space of the second modality. Multimodal embeddings are obtained in the following manner:
* When the two modalities are available, both are presented at their respective inputs and the activations are propagated through the network. The multimodal embedding is then obtained by concatenating the outputs of the middle layer.
* When one modality is available and the other is not, the available modality is presented to its respective input of the network and the activations are propagated. The central layer is then used to generate an embedding by being duplicated, thus still generating an embedding of the same size while allowing to transparently compare video segments regardless of modality availability.

Usage
=====
This code can either be used as a stand-alone tool or as a python module within other code.

As a stand-alone tool
---------------------

The following command line parameters are available:
```
$ ./bidnn.py --help

usage: bidnn.py [-h] [-a ACTIVATION] [-e EPOCHS] [-d DROPOUT] [-b BATCH_SIZE]
                [-r LEARNING_RATE] [-m MOMENTUM] [-l LOAD_MODEL]
                [-s SAVE_MODEL] [-c] [-n] [-z] [-u] [-w WRITE_AFTER]
                [-x EXEC_COMMAND] [-v VERBOSITY]
                infile outfile mod1size mod2size hdnsize repsize

positional arguments:
  infile                input file containing data in libsvm format
  outfile               output file where the multimodal representation is
                        saved in libsvm format
  mod1size              dimensionality of 1st modality
  mod2size              dimensionality of 2nd modality
  hdnsize               dimensionality of 1st hidden layer
  repsize               output (multimodal) representation dimensionality (2
                        * 2nd hdn layer dim)

optional arguments:
  -h, --help            show this help message and exit
  -a ACTIVATION, --activation ACTIVATION
                        activation function (default: tanh)
  -e EPOCHS, --epochs EPOCHS
                        epochs to train (0 to skip training and solely
                        predict)
  -d DROPOUT, --dropout DROPOUT
                        dropout value (default: 0.2; 0 -> no dropout)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        batch size (default: 128)
  -r LEARNING_RATE, --learning-rate LEARNING_RATE
                        learning rate (default: 0.1)
  -m MOMENTUM, --momentum MOMENTUM
                        momentum (default: 0.9)
  -l LOAD_MODEL, --load-model LOAD_MODEL
                        load pretrained model from file
  -s SAVE_MODEL, --save-model SAVE_MODEL
                        save trained model to file (at the end or after
                        WRITE_AFTER epochs); any %e will be replaced with the
                        current epoch
  -c, --crossmodal      perform crossmodal expansion (fill in missing
                        modalities) instead of multimodal embedding
  -n, --l2-norm         L2 normalize output (representation or reconstruction)
  -z, --ignore-zeroes   do not treat zero vectors as missing modalities
  -u, --untied          do not tie the variables in the central part
  -w WRITE_AFTER, --write-after WRITE_AFTER
                        write prediction file after x epochs (not only in the
                        end); use %e in [outfile] to indicate epoch number
  -x EXEC_COMMAND, --exec-command EXEC_COMMAND
                        execute command after WRITE_AFTER epochs (-w) or after
                        training; any %e in the command will be replaced with
                        the current epoch
  -v VERBOSITY, --verbosity VERBOSITY
                        sets verbosity (0-3, default: 3)
```

For example, to obtain a multimodal embedding of two modalities (e.g. text with dimensionality 100 and CNN visual features or dimensionality 4096) stored in a *dataset.txt*, the command could be:
```
./bidnn.py -n -e 10000 dataset.txt output.txt 100 4096 2048 2048
```
This would train a BiDNN with the first hidden layer with 2048 nodes and a representation size of 2048 (the 2nd and central hidden layer would be of size 2014 for each crossmodal translation).

To do the same but store the model, generate a prediction and run an external evaluation tool every 100 epochs, the command would be the following:
```
./bidnn.py dataset.txt output_%e.txt 100 4096 2048 2048 -n -e 10000 -w 100 -s model_%e.npz -x "./evaulate.py output_%e.txt"
```
####Input format
Input and output files are stored in LibSVM format:
```
0 1:val1 2:val2 3:val3, ... n:valn
```
The first number is a label (since this is unsupervised learning, it's ignored but it's still preserved in case the user is doing unsupervised learning with label data and needs the labeled preserved). Following are i:val pairs indicating each nonzero float value of the vectors representing each modality. If the two modalities are e.g. of dimensionality 100 and 300, each line would consist of 400 entries, first of the first modality, followed by the second modality (as defined in the mod1size and mod2size command line arguments).

As a Python module
-------------------
```python
from biddn import BiDNN, Config

conf = Config()
conf.mod1size = 100
conf.mod2size = 4096
conf.hdn = 2048
conf.rep = 2048

X = ... # load or generate data

bidnn = BiDNN(conf)
bidnn.load_dataset(X)

bidnn.train()
out = bidnn.predict()
```
Fore more details, you can take a look at the *\__main\__* section of *bidnn.py*.

Requirements
============

* Theano
* Lasagne (tested under version 0.2.dev1)
* NumPy
* SciPy (solely for L2 normalization)
* scikit-learn (solely to load and write LIBSVM formatted files - not needed if used as a python module)

Citing
======

If you used and/or liked this tool, please consider citing the original paper where the methodology is described:
```
@InProceedings{vukotic-icmr-16,
  title={Bidirectional Joint Representation Learning with Symmetrical Deep Neural Networks for Multimodal and Crossmodal Applications},
  author={Vukoti{\'c}, Vedran and Raymond, Christian and Gravier, Guillaume},
  booktitle={Proceedings of the 2016 ACM International Conference on Multimedia Retrieval (ICMR)},
  pages={343--346},
  year={2016},
  organization={ACM}
}
```
and/or a paper with additional evaluation and analysis in video hyperlinking:
```
@inproceedings{vukotic-ivlmm-16,
  title = {{Multimodal and Crossmodal Representation Learning from Textual and Visual Features with Bidirectional Deep Neural Networks for Video Hyperlinking}},
  author = {Vukotic, Vedran and Raymond, Christian and Gravier, Guillaume},
  booktitle = {{ACM Multimedia 2016 Workshop: Vision and Language Integration Meets Multimedia Fusion (iV\&L-MM'16)}},
  address = {Amsterdam, Netherlands},
  organization = {{ACM}},
  year = {2016},
  month = Oct
}
```
