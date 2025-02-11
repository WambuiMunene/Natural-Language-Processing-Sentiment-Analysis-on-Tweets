# Phase-4-Sentiment-Analysis-NLP-Project

## Project Summary

Social media is a dynamic platform where customers express their thoughts about products, services, and brands. Analyzing sentiments from social media platforms like X (formerly Twitter) provides businesses with real-time insights into customer opinions and experiences.

## Data Understanding

The objective of this project is to build a Natural Language Processing (NLP) model that rates the sentiment of tweets about Apple and Google products as positive, negative or neutral. The dataset used to build the model is sourced from CrowdFlower via data.world https://data.world/crowdflower/brands-and-product-emotions. This dataset consists of slightly over 9,000 human-rated tweets.

Features: prior to the preprocessing steps every row in the dataset only contains two feature columns; a string containing the full text of an individual tweet, and another string on the product being refereed to in the tweet. During preprocessing a string of tweet text will be converted inoto individual words creating more features.
Target: the target consists of labels (emotions) for different tweets - positive, negative, neutral and 'can't tell'. By looking at the value counts for each sentiment, a decision will be made on which of the classes to use to achieve our objectives

## Problem Statement

Sentiment Analysis provides businesses with insights into public perception of their products and services. By analyzing sentiments from tweets, companies can identify areas of concern in real-time, allowing them to address customer needs proactively.

## Business Objectives

1.	**Goal:** Train classification models to identify sentiments (Positive, Neutral, Negative) about Apple and Google Products.
2.	**Specific Objectives:**
 - Identify the distribution of negative and positive tweets by company.
 - Train, tune, and evaluate at least 3 classification models for sentiment analysis.
 - Provide the optimal model to Apple for identifying negative sentiments in future data.

 ## Requirements to Meet Objectives

1. Load the Data
•	- Use Pandas to load the dataset and inspect the data.

2. Perform Data Cleaning with nltk
•	- Use Regular Expressions (REGEX) to remove irrelevant information such as URLs, mentions, and hashtags.
•	- Convert all text to lowercase to ensure uniformity.
•	- Apply lemmatization to reduce words to their base forms.
•	- Remove stop words to focus on meaningful words.
•	- Tokenize the cleaned text.
3. Perform Exploratory Data Analysis
•	- Analyze positive and negative sentiments by company.
•	- Visualize the distribution of sentiment labels using bar charts and value counts.
•	- Visualize the top 10 most common words.
•	- Create word clouds for positive, negative, and neutral tweets.

4. Vectorize the Text Data with TFidfVectorizer
•	- Use TF-IDF vectorizer  to convert the text data into numeric form.

5. Iteratively Build and Evaluate Baseline and Ensemble Models
•	- Use Pipelines to build and tune Logistic Regression and Naive Bayes Models.
•	- Build and train one or more ensemble models and compare results with tuned baseline models.

6. Evaluation
•  Evaluate model performance using:
o	- classification_report from Scikit-learn
o	- confusion_matrix

## Libraries Used

Libraries Used
1. Pandas
•	Purpose: Data manipulation and analysis.
•	Usage: Loading datasets, cleaning data, and transforming data for analysis.

2. nltk (Natural Language Toolkit)
•	Purpose: Text preprocessing.
•	Usage: Tokenization, lemmatization, removing stop words, and text cleaning.

3. scikit-learn (sklearn)
•	Purpose: Machine learning and model evaluation.
•	Usage: Building and evaluating models, including Logistic Regression, Naive Bayes, and ensemble models. Metrics such as classification_report and confusion_matrix.

4. matplotlib ans seaborn
•	Purpose: Data visualization.
•	Usage: Creating bar charts, value counts, and other visualizations to understand class balance.

5. wordcloud
•	Purpose: Text visualization.
•	Usage: Creating word clouds to visualize the most common words in each sentiment class.

6. regex
•	Purpose: Text cleaning.
•	Usage: Handling regular expressions for removing irrelevant information such as URLs, mentions, and hashtags.

## Next Steps and Recommendations

### Recommendations

1. Sentiment Analysis and Competition Landscape

![Sentiments By Product](sentiments_by_product.png)

•	Popularity and Sentiment Balance: Apple products are more popular but also have higher negative sentiments. Apple should monitor and address negative sentiments to maintain its market position.
•	Strategy for Negative Sentiments: Apple should proactively address customer complaints by enhancing customer service, improving product quality, and engaging with users on social media.
3.2 Model Performance and Selection
Evaluation of Sentiment Classification Models to Identify Positive, Neutral, and Negative Classes
•	Baseline Logistic Regression Model showed an overall accuracy of 65%, while the tuned Logistic Regression and Random Forest Model had an accuracy of 66% and 67% respectively. Comparing the metrics for the negative class:
o	Precision: Random Forest is better at avoiding false positives for negative tweets.
o	Recall: Baseline Logistic Regression captures a higher percentage of actual negative tweets.
o	F1-Score: Baseline Logistic Regression offers a balanced approach with better recall.
The Baseline Logistic Model is the better model for identifying the three classes, focusing on improving the recall of the Negative Class.
Evaluation of Sentiment Classification Models to Identify the Negative Class
•	Sub-optimal performance can be attributed to class imbalance. Although SMOTE was used to oversample the minority class, the synthetic data did not significantly enhance model performance.
•	Steps taken:
o	Class Consolidation: Neutral and Positive classes combined into a new class labeled 'Other'.
o	Resampling: Built a model with a resampled subset of the new class.
o	Model Training: Trained both baseline and tuned Logistic Regression models, along with three Ensemble models.
Given the focus on identifying negative sentiments accurately, the Baseline Logistic Regression model with balanced classes is recommended. It achieves the highest recall for the negative class, ensuring a higher number of negative sentiments are accurately identified.




