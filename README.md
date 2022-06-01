# Meta Pseudo Labels
This is an unofficial PyTorch implementation of [Meta Pseudo Labels](https://arxiv.org/abs/2003.10580) with some modification for general supervised learning (Binary classification).
The official Tensorflow implementation is [here](https://github.com/google-research/google-research/tree/master/meta_pseudo_labels).
unofficial Pytorch implementation

# Focusing
this is a function to focus learning mistakes from student test using seen samples.
in this implementation, the modification is combining mistakes from student and mistakes from teacher.

teacher loss = (loss teacher in seen samples) + (weight_unseen * loss student in seen samples)

# Semi Supervised
the idea of semi supervised is to create minimal label data and improve the model with a-lot of non label data.
From what I learn, we only need to label some key samples from data as seen samples.
these seen samples are better to have characteristic that different with each other in seen data.
if we know key points that representative enough to whole data we can have as small labeled samples as possible.  

for example:

![alt text](http://url/to/img.png)
