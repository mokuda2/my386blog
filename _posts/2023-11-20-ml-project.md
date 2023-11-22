---
layout: post
title: Pokémon Type Predictions with Base Stats and Image Classification
author: Michael Okuda
description: Is there a way to predict the type of a Pokémon based on its base stats and image?
image: /assets/images/pokeball.png
---

## Introduction

The strategy to beating Pokémon is to figure out the type of the Pokémon and then use moves that are super effective against it.  Once a player knows what types are effective, the concept of winning Pokémon battles is pretty easy.

What patterns can a player find to predict the type of a Pokémon he has never seen before?  One way is to make predictions based off its name.  For example, "Beautifly" is probably a butterfly, and butterflies are bugs and can fly.  Therefore, a player might conclude that Beautifly are Bug/Flying type, which is correct.  The methods I will use to predict a Pokémon's type are base stat ratios and image classification.

## Base Stat Ratios

Every Pokémon has six base stats: hit points (HP), attack, defense, special attack, special defense, and speed.