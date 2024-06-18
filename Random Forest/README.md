## Understanding Random Forests

- works when you don't have lot of training data

  - when there is sparse data

- Collection of a bunch of decision trees
- SUPERVISED : need labelled data

Decision Tree:

    - root should give the best split of input data
    - based on Gini impurity (to calculate the splits)

    Disadvantages:
        - Overfitting
        - fails on new data

## RANDOM FOREST TO THE RESCUE

Random Forest: - collection of decision trees

Steps:

    1. Bootstrapped datasets (RANDOMLY)
        - contains 2/3rd of the input data
        - some input data gets repeated

    2. Making decision trees
        - Selecting root, picks one from random subset of features available with highest information gain

Validation/ Verification:

    - dropped 1/3rd of the data run by the RF tree and see the accuracy!

## Feature Importance:

    - which is the best factor that is contributing? RF does!
    - can list feature_names
