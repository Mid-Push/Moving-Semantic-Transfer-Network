# Moving-Semantic-Transfer-Network
## Usage: python smtrain.py
Or u could # mkdir data # and put your own SVHN and MNIST in there directly.


Paper is available at http://shaoan.vision/publications/icml2018.pdf 


For Digit datasets, I modify the codes from https://github.com/erictzeng/mldata

For network, I use https://github.com/dgurkaynak/tensorflow-cnn-finetune

# Deep Transfer Learning on Caffe

This is a caffe library for deep transfer learning. We fork the repository with version ID `29cdee7` from [Caffe](https://github.com/BVLC/caffe) and make our modifications. The main modifications are listed as follow:

- Add `mmd layer` described in paper "Learning Transferable Features with Deep Adaptation Networks" (ICML '15).
- Add `jmmd layer` described in paper "Deep Transfer Learning with Joint Adaptation Networks" (ICML '17).
- Add `entropy layer` and `outerproduct layer` described in paper "Unsupervised Domain Adaptation with Residual Transfer Networks" (NIPS '16).
- Copy `grl layer` and `messenger.hpp` from repository [Caffe](https://github.com/ddtm/caffe/tree/grl).
- Emit `SOLVER_ITER_CHANGE` message in `solver.cpp` when `iter_` changes.

Data Preparation
---------------
In `data/office/*.txt`, we give the lists of three domains in [Office](https://cs.stanford.edu/~jhoffman/domainadapt/#datasets_code) dataset.

We have published the Image-Clef dataset we use [here](https://drive.google.com/file/d/0B9kJH0-rJ2uRS3JILThaQXJhQlk/view?usp=sharing).

Training Model
---------------

In `models/DAN/alexnet`, we give an example model based on Alexnet to show how to transfer from `amazon` to `webcam`. In this model, we insert mmd layers after fc7 and fc8 individually.


In `models/RTN/alexnet`, we give an example model based on Alexnet to show how to transfer from `amazon` to `webcam`. In this model, we insert mmd layers after the outer product of the outputs of fc7 and fc8.

In `models/JAN/alexnet` and `models/JAN/resnet`, we give an example model based on Alexnet and ResNet respectively to show how to transfer from `amazon` to `webcam`. In this model, we insert jmmd layers with outputs of fc7 and fc8 as its input.

The [bvlc\_reference\_caffenet](http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel) is used as the pre-trained model for Alexnet. The [deep-residual-networks](https://github.com/KaimingHe/deep-residual-networks) is used as the pre-trained model for Resnet. We use Resnet-50. If the Office dataset and pre-trained caffemodel are prepared, the example can be run with the following command:
```
Alexnet:

"./build/tools/caffe train -solver models/*/alexnet/solver.prototxt -weights models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel (*=DAN, RTN or JAN)"
```
```
ResNet:

"./build/tools/caffe train -solver models/JAN/resnet/solver.prototxt -weights models/deep-residual-networks/ResNet-50-model.caffemodel"
```



## Citation
If you use this library for your research, we would be pleased if you cite the following papers:

```
@inproceedings{xie2018learning,
  title={Learning Semantic Representations for Unsupervised Domain Adaptation},
  author={Xie, Shaoan and Zheng, Zibin and Chen, Liang and Chen, Chuan},
  booktitle={International Conference on Machine Learning},
  pages={5419--5428},
  year={2018}
}
```

## Contact
If you have any problem about this library, please create an Issue or send us an Email at:
- shaoanxie@outlook.com
