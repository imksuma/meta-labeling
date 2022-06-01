# Meta Pseudo Labels
This is an unofficial implementation of [Meta Pseudo Labels](https://arxiv.org/abs/2003.10580) with some modification for general supervised learning (Binary classification).
The official Tensorflow implementation is [here](https://github.com/google-research/google-research/tree/master/meta_pseudo_labels).
the exact unofficial Pytorch implementation from [paper](https://github.com/google-research/google-research/tree/master/meta_pseudo_labels) is [here](https://github.com/kekmodel/MPL-pytorch).

# Focusing
this is a function focus to learning mistakes from student testing by seen samples.
in this implementation, teacher loss is combination loss from student and loss from seen samples.

teacher loss = (loss teacher in seen samples) + (weight_unseen * loss student in seen samples)

# Semi Supervised
the idea of semi supervised is to create minimal label data and improve the model with a-lot of non-labeled data.
From what I learn, we only need to label some key samples from data as seen samples.
these seen samples are better to have characteristic that different with each other in seen data.
if we know key points that representative enough of the whole data, we can have as small labeled samples as possible.  

for example:

![inference result meta labeling](https://github.com/imksuma/meta-labeling/blob/main/inference-meta-label.png)

we randomly select but within criteria that represent enough for two moon dataset.
from running several experiment, we have consistent good performance using meta label compare to supevised learning.

## Supervised can be good using minimal representative sample
from running experiments (two moon) for some attempts,
supervised learning with minimal samples, that have criteria of enough representation of the whole data,
can also have good performance.