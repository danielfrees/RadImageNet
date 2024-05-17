PyTorch v3 had fixes to Caffe preprocessing, train dataloader shuffling (especailly important for ACL), and a handful of other fixes. 

PyTorch v4 architecture removes the softmax from the classifier appended to the backbone, relying instead on SoftmaxLoss so that we don't do a double softmax. This massively improves breast performance. 

