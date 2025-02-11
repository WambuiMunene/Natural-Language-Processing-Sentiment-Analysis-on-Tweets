# Phase-4-Sentiment-Analysis-NLP-Project
## Project Summary
Social media is a dynamic platform where customers express their thoughts about products, services, and brands. Analyzing sentiments from social media platforms like X (formerly Twitter) provides businesses with real-time insights into customer opinions and experiences.

## Data Understanding
The objective of this project is to build a Natural Language Processing (NLP) model that rates the sentiment of tweets about Apple and Google products as positive, negative or neutral. The dataset used to build the model is sourced from CrowdFlower via data.world https://data.world/crowdflower/brands-and-product-emotions. This dataset consists of slightly over 9,000 human-rated tweets.

Features: prior to the preprocessing steps every row in the dataset only contains two feature columns; a string containing the full text of an individual tweet, and another string on the product being refereed to in the tweet. During preprocessing a string of tweet text will be converted inoto individual words creating more features.
Target: the target consists of labels (emotions) for different tweets - positive, negative, neutral and 'can't tell'. By looking at the value counts for each sentiment, a decision will be made on which of the classes to use to achieve our objectives

Problem Statement
Sentiment Analysis provides businesses with insights into public perception of their products and services. By analyzing sentiments from tweets, companies can identify areas of concern in real-time, allowing them to address customer needs proactively.
Business Objectives
1.	Goal: Train classification models to identify sentiments (Positive, Neutral, Negative) about Apple and Google Products.
2.	Specific Objectives:
 - Identify the distribution of negative and positive tweets by company.
 - Train, tune, and evaluate at least 3 classification models for sentiment analysis.
 - Provide the optimal model to Apple for identifying negative sentiments in future data.

