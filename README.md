# Moving-Semantic-Transfer-Network



Tensorflow Implementation for moving semantic transfer network (ICML2018).

<img src="introduction/mstn_network.PNG" width=400 />

Based on adversarial adaptation, we propose a `Pseudo Centroid Alignment Objective` to enforce `Semantic Transfer`.

## Citation
If you find this useful for your research, we would be pleased if you cite the following papers:

```
@inproceedings{xie2018learning,
  title={Learning Semantic Representations for Unsupervised Domain Adaptation},
  author={Xie, Shaoan and Zheng, Zibin and Chen, Liang and Chen, Chuan},
  booktitle={International Conference on Machine Learning},
  pages={5419--5428},
  year={2018}
}
```

## Tips for your reproduction of our work.

My work is based on DANN. During my reimplementation of DANN, I noticed following problems worth attention for reproduing DANN and our work MSTN. Hope these could help you. :)

<li> Data Preprocessing </li>
<ol type="a">
<li> Scale image to 256x256. </li> 
<li >When training, source and target images would be "Random Cropping". When testing, target images would be "Center Cropping" (Caffe only uses crop command but the inner implemenattion actually random crop when training while center crop when testing). </li>
</ol>

<li> Hyperparameter </li>



## Contact
If you have any problem about this library, please create an Issue or send us an Email at:
- shaoanxie@outlook.com
