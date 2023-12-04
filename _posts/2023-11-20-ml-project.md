---
layout: post
title: Pokémon Type Predictions with Base Stats and Image Classification
author: Michael Okuda
description: Is there a way to predict the type of a Pokémon based on its base stats and image?
image: /assets/images/pokeball.png
---

## Introduction

The strategy to beating Pokémon is to figure out the type of the Pokémon and then use moves that are super effective against it.  Once a player knows what types are effective, the concept of winning Pokémon battles is pretty easy.

What patterns can a player find to predict the type of a Pokémon he has never seen before?  One way is to make predictions based off its name.  For example, "Beautifly" is probably a butterfly, and butterflies are bugs and can fly.  Therefore, a player might conclude that Beautifly are Bug/Flying type, which is correct.  The methods I will use to predict a Pokémon's type are base stat ratios and image classification.  The reason for exploring multiple models is that no single model performs well in every situation.  Therefore, it is good to implement different models.  The data I used for the analysis are from the [PokeAPI website](https://pokeapi.co/docs/v2#pokemon) and a [Kaggle dataset of Pokémon images](https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types).

## Exploratory Data Analysis (EDA)

[This article](https://medium.com/m2mtechconnect/classifying-pok%C3%A9mon-images-with-machine-learning-79b9bc07c080) has some great approaches to EDA for the image classification dataset.  I really liked the visualization of the different types for each Pokémon.  For simplification, only the first type is used to classify the type of a Pokémon.  According to the plot, water types are most common in the Kaggle dataset, while flying is the least common.

![Figure](https://raw.githubusercontent.com/mokuda2/my386blog/main/assets/images/number-of-types.png)

From the data scraped off the PokeAPI website, any Pokémon game player knows that evolved forms of Pokémon have higher base stats than their pre-evolved forms.  To avoid the issue of the wide ranges of base stats among a Pokémon and its evolved forms, I used ratios of base stats.  For simplicity, I divided all base stats by speed except for speed itself.  In a way, it "standardizes" the base stats.  For example, Charmander has base stats of 52 for attack and 65 for speed, while its most evolved form Charizard has base stats of 84 for attack and 100 for speed.  Charmander's base stats are much lower than Charizard's, even though they are both Fire type.  Attack divided by speed gives Charmander a .80 ratio and Charizard a .84 ratio, which are much more similar.

You can also take a look at my [EDA blog post](https://mokuda2.github.io/my386blog/2023/03/26/eda.html) for more insights into the PokeAPI website's data.

## Overview of Models Tried

Below are the following models I used for the image classification dataset:

* convolutional neural network: a neural network especially designed for recognizing patterns in image data.  Hyperparameters I used include the activation function, input shape, and number of hidden layers.  When evaluating the model on predicting from all 18 types, the testing accuracy is only .14.  When evaluating the model on just two types, specifically Bug vs. Fire, the testing accuracy is .71.
* feed forward neural network: a neural network that only goes forward and doesn't back propagate.  Hyperparameters I used include the activation function, input shape, and number of hidden layers.  When evaluating the model on predicting from all 18 types, the testing accuracy is only .21.  When evaluating the model on just two types, specifically Bug vs. Fire, the testing accuracy is .68.

Below are the following models I used for the PokeAPI dataset:

* k-nearest neighbors: classifies a new data point based on a distance metric from the k closest data points.  Hyperparameters include the number of neighbors (k) and the distance metric.  When evaluating the model on predicting from all 18 types, the testing accuracy is .23.  When evaluating the model on just two types, specifically Bug vs. Fire, the testing accuracy is .8.
* decision tree: classifies an instance by looking at conditions of the nodes until it reaches the bottom.  Hyperparameters include the depth of the tree and the criteria for measuring what feature should be used on the split.  When evaluating the model on predicting from all 18 types, the testing accuracy is .18.  When evaluating the model on just two types, specifically Bug vs. Fire, the testing accuracy is .7.
* random forest: creates several decision trees through bagging and uses a subset of features to determine the branching off of each node.  Hyperparameters include the depth of the tree and the maximum number of features to consider when splitting.  When evaluating the model on predicting from all 18 types, the testing accuracy is .27.  When evaluating the model on just two types, specifically Bug vs. Fire, the testing accuracy is 86.
* ensemble model: a model that uses several models to improve performance.  I used the KNN, decision tree, and random forest models to create the ensemble model.  Hyperparameters I used include the number of neighbors for the KNN model, the maximum depth of the tree for decision tree model, and the number of features to determine each split and the maximum depth of each tree for the random forest model.  When evaluating the model on predicting from all 18 types, the testing accuracy is .26.  When evaluating the model on just two types, specifically Bug vs. Fire, the testing accuracy is .73.
* k-means clustering: assigns a data point to a cluster based on how close it is to other clusters.  The difference between KNN and clustering is that KNN has labeled data points for classification, whereas clustering simply groups data points together based on patterns and not on labels.  One important hyperparameter is the number of clusters.  When evaluating the model on predicting from all 18 types, the testing accuracy is .21.  When evaluating the model on just two types, specifically Bug vs. Fire, the testing accuracy is .55.
* PCA: dimension reduction technique that reduces the features of a dataset while preserving as much variance in the data as possible.

## Discussion on Model Selection

KNN's testing performance is better than decision trees but worse than random forest when classifying between all 18 types.  Surprisingly, the ensemble method's testing performance is slightly worse than random forest.  One challenge faced with the decision tree model is that it can easily overfit if the maximum depth of the tree isn't specified.  For the decision tree's training accuracy, it is .98, but its testing accuracy is only .18, which is a sign of overfitting.  Similarly, for the image classification models with all 18 types, training accuracy is fairly high, but validation accuracy is super low.

For binary classification between Fire type and Bug type, training and testing accuracies are fairly high and generally close to each other for both the images dataset and the PokeAPI dataset.  However, I think decision trees still overfit when the maximum depth of the tree isn't tuned.  I think that decision tree models can be cut out, as random forests give much better metrics and make up several decision trees.  I also prefer using the binary classification models, as their predictions are a lot more accurate.  A Pokémon player could probably take a guess between two different types when looking at a Pokémon.

## Detailed Discussion on Best Model

For the images dataset for binary classification, the model that performs better is the convolutional neural network.  Compared to the feed forward neural network, it makes sense because CNNs are typically used for image data, and FFNs only go forward in their neural networks.  One of the big differences between tuning a CNN model with a target variable with 18 outcomes and a CNN model with a binary target variable is that the last layer for the binary classification can be sigmoid, which is what logistic regression models use.  The binary classification model metrics for training and validation accuracies are shown below for the CNN model.  The testing accuracy is .71.

![Figure](https://raw.githubusercontent.com/mokuda2/my386blog/main/assets/images/cnn-graph.png)

While neural networks can be accurate, it is a lot more difficult in terms of interpretability.  Tuning hyperparameters, such as the number of hidden layers or the activation functions, can help improve model performance, though it is hard to explain why a hyperparameter's value is best for the situation.

For the PokeAPI dataset for binary classification between Fire type and Bug type, the model that performs best is KNN, which is surprising.  Below are metrics for accuracy, precision, and recall for the KNN, random forest, and ensemble models.

![Figure](https://raw.githubusercontent.com/mokuda2/my386blog/main/assets/images/metrics-for-three-models.png)

Below are the ROC curves for the KNN, random forest, and ensemble models, respectively:

![Figure](https://raw.githubusercontent.com/mokuda2/my386blog/main/assets/images/knn-roc.png)

![Figure](https://raw.githubusercontent.com/mokuda2/my386blog/main/assets/images/random-forest-roc.png)

![Figure](https://raw.githubusercontent.com/mokuda2/my386blog/main/assets/images/ensemble-roc.png)

For KNN, I left the hyperparameters at default, such as k=5.  The reason I'm surprised that the KNN model did the best is that the random forest and ensemble models are diverse in their ways of aggregating several predictions and models in hopes of improving performance.  For the random forest model, I did a grid search for the arguments of criterion, max_depth, max_features, and n_estimators.  For the ensemble model, I also did a grid search for the arguments of KNN's n_neighbors, decision tree's max_depth, and random forest's max_depth and n_estimators.  While random train-test splits of the data may improve the metrics of the random forest and ensemble models, the KNN model's performance shows that no single model can outperform all others in every situation, even random forests and ensemble methods.

As far as the interpretability of the KNN model, the default value for k is 5, meaning that a new data point will find the five closest data points in terms of a distance metric.  A majority vote is used to determine the prediction of the new data point.  For example, if two data points are Fire type and the other three data points are Bug type, then the new data point will be classified as Bug type.  Certain ratios of base stats may be important in the prediction of one type or the other.  Fire vs. Bug is only one of several binary combinations, so the models' performances may be different if, say, we do binary classification for Water vs. Psychic.

## Conclusion and Next Steps

To reiterate, the best model from the image classification dataset is the convolutional neural network model, and the best model from the PokeAPI dataset is the KNN model.  CNNs can perform well on image data, so that model works well for the Pokémon images.  The KNN's model performance trumping the other models shows that even a simple model can be a great model.

As far as potential future steps, I would do binary classification for all combinations of two types and compare the models' metrics.  Maybe the model performances will be different, and the random forest model or the ensemble model may outperform KNN.  I also think it would be interesting to figure out a way for a model to take the two types that have the highest probability for being predicted and then running the model again on just those two types.  That way, the model may have better prediction metrics; the common theme is that binary classification models perform a lot better than models that try to predict one of the 18 types.