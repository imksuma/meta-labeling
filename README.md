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

# Imbalance case
From my experience in real world problem, the industry might not have much time and resource to label data or
to have valid data from user report. for example, consider problem of Account Take over, typically the industry
will have some report system that recieve complaint from customer. the customer will report the incident of account take over.
thus, the company will label the account as ATO along with date of ATO login (typically might not be the exact date, and worse not include the exact 
second, since the customer only know the problem when their money is gone but not when the fraudster login).
thus this project will be an imbalnce case. and the worst part is undetected ATO login due to the customer is not making a report.

and the question :
## Can we handle typical Imbalance case like this using meta label?
It depend, when `key point` of positive samples are good enough we can make the `seen data` 1:4 positive:negative ratio or even close to 1:2 ratio.
with 1 as imbalance positive and 4 as negative. and the majority of data can be treat as unseen samples.