# DL_Siamese_Network_MNIST_Classification
This repository contains my implementation of Siamese Network, which purpose is MNIST digits classification.

Siamese Networks are designed to be especially good at solving *one shot classification* tasks. The Siamese architecture consists of two identical networks, whose weights
are share, connected by their heads (just like siamese twins). The workflow is as follows:
1. Feed both networks with two different images.
2. Compute *contrastive loss* (more about it later).
3. Backpropagate loss through both networks.

The network's purpose is not to directly classify things, but rather to predict, wheter images belong to the same class. That is the *contrastive loss* for.


