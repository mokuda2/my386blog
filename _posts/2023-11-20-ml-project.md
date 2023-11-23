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

From the data scraped off the PokeAPI website, any Pokémon game player knows that evolved forms of Pokémon have higher base stats than their pre-evolved forms.  To avoid the issue of the wide ranges of base stats among a Pokémon and its evolved forms, I used ratios of base stats.  For simplicity, I divided all base stats by speed except for speed itself.  In a way, it "standardizes" the base stats.  For example, Charmander has base stats of 52 for attack and 65 for speed, while its most evolved form Charizard has base stats of 84 for attack and 100 for speed.  Charmander's base stats are much lower than Charizard's, even though they are both fire type.  Attack divided by speed gives Charmander a .80 ratio and Charizard a .84 ratio, which are much more similar.

You can also take a look at my [EDA blog post](https://mokuda2.github.io/my386blog/2023/03/26/eda.html) for more insights into the PokeAPI website's data.

## Overview of Models Tried

Below are the following models I used for the image classification dataset:

* convolutional neural network: a neural network especially designed for recognizing patterns in image data.  Hyperparameters I used include the activation function, input shape, and number of hidden layers.  When evaluating the model on predicting from all 18 types, the testing accuracy is only .14.  When evaluating the model on just two types, specifically bug vs. fire, the testing accuracy is .71.
* feed forward neural network: a neural network that only goes forward and doesn't back propagate.  Hyperparameters I used include the activation function, input shape, and number of hidden layers.  When evaluating the model on predicting from all 18 types, the testing accuracy is only .21.  When evaluating the model on just two types, specifically bug vs. fire, the testing accuracy is .68.

Below are the following models I used for the PokeAPI dataset:

* k-nearest neighbors: classifies a new data point based on a distance metric from the k closest data points.  Hyperparameters include the number of neighbors (k) and the distance metric.  When evaluating the model on predicting from all 18 types, the testing accuracy is .23.  When evaluating the model on just two types, specifically bug vs. fire, the testing accuracy is .8.
* decision tree: classifies an instance by looking at conditions of the nodes until it reaches the bottom.  Hyperparameters include the depth of the tree and the criteria for measuring what feature should be used on the split.  When evaluating the model on predicting from all 18 types, the testing accuracy is .18.  When evaluating the model on just two types, specifically bug vs. fire, the testing accuracy is .
* random forest: creates several decision trees through bagging and uses a subset of features to determine the branching off of each node.  Hyperparameters include the depth of the tree and the maximum number of features to consider when splitting.  When evaluating the model on predicting from all 18 types, the testing accuracy is .27.  When evaluating the model on just two types, specifically bug vs. fire, the testing accuracy is .
* ensemble model: a model that uses several models to improve performance.  I used the KNN, decision tree, and random forest models to create the ensemble model.  Hyperparameters I used include the number of neighbors for the KNN model, the maximum depth of the tree for decision tree model, and the number of features to determine each split and the maximum depth of each tree for the random forest model.  When evaluating the model on predicting from all 18 types, the testing accuracy is .26.  When evaluating the model on just two types, specifically bug vs. fire, the testing accuracy is .
* k-means clustering: assigns a data point to a cluster based on how close it is to other clusters.  The difference between KNN and clustering is that KNN has labeled data points for classification, whereas clustering simply groups data points together based on patterns and not on labels.  One important hyperparameter is the number of clusters.