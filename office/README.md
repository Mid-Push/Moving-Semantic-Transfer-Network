
# Moving-Semantic-Transfer-Network

## [Click here for the trained MSTN model](https://drive.google.com/drive/folders/1o10GWduF3QI7p55x14YwyYxykjEPi8Jz?usp=sharing).

>>Pretrained Alexnet model    ------ bvlc_alexnet.npy

>>trained MSTN model          ------ 10000.ckpt


## Training Model

Download the pretrained Alexnet Model and add the model to `alexnet` directory.

Download the [Office-31](https://cs.stanford.edu/~jhoffman/domainadapt/#datasets_code) or [ImageCLEF](https://drive.google.com/file/d/0B9kJH0-rJ2uRS3JILThaQXJhQlk/view?usp=sharing) and put the images in `Home` directory just like this.
```/home/dataset/office/domain_adaptation_images/amazon/images/calculator/frame_0001.jpg 5```

Choose the transfer task in `mstntrain.py` (default is A->W) and 

```
Alexnet:

"./build/tools/caffe train -solver models/*/alexnet/solver.prototxt -weights models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel (*=DAN, RTN or JAN)"
```
.
