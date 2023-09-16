#@ Implementation of triplet loss
import tensorflow as tf
margin  = ...                                                                # hyperparameter margin between postive and negative
Anchor_output = ...                                                          # embedding of anchor
Positive_output = ...                                                        # embedding of positive i.e similar
Negative_output = ...                                                        # embedding of negative i.e dissimilar

d_pos = tf.reduce_sum(tf.square(Anchor_output - Positive_output), 1)         # eucilidiean distance of anchor and postive embedding
d_neg = tf.reduce_sum(tf.square(Anchor_output - Negative_output), 1)         # eucilidiean distance of anchor and negative embedding

loss = tf.maximum(d_pos-d_neg+ margin, 0.0)                                  # triplet loss

loss = tf.reduce_mean(loss)