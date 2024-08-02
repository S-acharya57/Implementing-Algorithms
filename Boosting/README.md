"""
Created on Wed Jun 26 10:15:24 2024

@author: sajjan
"""

## Boosting

- ensemble in sequence
  - i.e setup of the tree is influenced by the previous tree models
  - some learning
  - $n^{th}$ tree has dependency on predictions of $(n-1)^{th} tree$

### AdaBoost:

    - handles outliers and label noise robustly
    - computationally fast

### GradientBoost

    - categorical and numerical features
    - large number of features is present!
