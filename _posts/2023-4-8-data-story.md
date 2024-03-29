---
layout: post
title:  The Story of the Pokémon Data
author: Michael Okuda
description: What is the story the data tells from my exploratory data analysis in my last blog post?
image: /assets/images/pokeball.png
---

## Introduction

In my [first post of this project](https://mokuda2.github.io/my386blog/2023/03/14/web-scraping-part-1.html), I web scraped Pokémon data involving their base stats.  In my [previous post](https://mokuda2.github.io/my386blog/2023/03/26/eda.html), I did exploratory data analysis to figure out which Pokémon would be the strongest to train.  In this blog post, I will share the story that the data is telling and the high-level takeaways.  This [GitHub repository](https://github.com/mokuda2/pokemon) contains the code I used to create the visualization in this blog post.

## The Story

<iframe src="{{site.url}}/{{site.baseurl}}/assets/images/base-stats-total-with-frequency-count.html" width="100%" height="500px"></iframe>

In the above visualization, I arranged the top 20 Pokémon with the highest base stats total in descending order.  The color of a bar represents the count of a Pokémon's name in the data frames of the highest base stats as well as the highest base stats total.

It is interesting to observe that although Hydreigon has the highest base stats total, it is listed in three of the data frames, whereas Serperior and Archeops are listed in four.  For instance, Serperior is listed in the top 20 for defense, special defense, speed, and base stats total.

For those who are not familiar with Pokémon games, this analysis may not be that exciting or interesting since Pokémon names may have no meaning whatsoever.  However, for those who are familiar with Pokémon games, here are the Pokémon I would choose to be part of my team:

* Archeops
* Haxorus
* Vanilluxe
* Emboar
* Klinklang
* Seismitoad

So why did I choose these Pokémon?  Why didn't I choose the top six from the above graph?  One reason is that Serperior, Emboar, and Samurott are the fully-evolved forms of the [starter Pokémon](https://bulbapedia.bulbagarden.net/wiki/Starter_Pok%C3%A9mon), and a player can choose only one of them.  After playing a few rounds of Pokémon White, I know what Pokémon appear at what point in the game and when they evolve.  For example, Hydreigon's pre-evolution evolves at level 64, which is usually not reached until after beating the game.  I also chose Seismitoad—even though it was towards the bottom of the graph—because it is almost essential to have a water-type Pokémon in order to cross water.  It does take some context to understand what Pokémon to choose from the list, but the insights of the data are helpful in making that decision.

## Conclusion

One thing I would have done to elaborate on this analysis is to include the type of the Pokémon and see which Pokémon had the highest base stats based on a certain type.  A [type](https://bulbapedia.bulbagarden.net/wiki/Type#:~:text=Types%20%28Japanese%3A%20%E3%82%BF%E3%82%A4%E3%83%97%20Type%29%20are%20properties%20applied%20to,I%2C%20types%20were%20occasionally%20referred%20to%20as%20elements.) is a property that Pokémon have as well as moves in battle.  Examples of types include fire, water, and grass, and some Pokémon may have two types.  A well-rounded Pokémon party includes a variety of types not only of the Pokémon themselves but also their moves.  For all you Pokémon fans, comment which Pokémon you'd choose to have in your party, and then go play!