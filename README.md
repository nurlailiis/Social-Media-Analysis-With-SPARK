# Social Media Analysis and Classification of Tweets Using SPARK
In this section I try to do project Twitter Sentiment Analysis. In this project, Sentiment Analysis Application is developed using Pyspark for only sneak peak the data, then for analysis, data preparation, visualization, modeling, and evaluation using Python. This application fetches Twitter data in live stream and classifies tweets into positive and negative categories. For sentiment classification of tweets, machine learning model (Voting Mechanism) has been developed. Spark’s ability to perform well on iterative algorithms makes it ideal for implementing machine learning techniques as, at their vast majority, machine learning algorithms are based on iterative jobs. Further, live visualization of results is done using Flask and Chart.js technology. Visualization gives the ability to combine data in order to create new insight.

## Get Started
1. The dataset using clean_tweet.csv file consist of data scraped from Twitter, contain text and target negative or positive
2. Analyze data using [CRISP-DM](https://www.sv-europe.com/crisp-dm-methodology/) methodology
    - Business Understanding
    - Data Understanding
    - Data Preparation : duplication detection, stopwords removal, rare words removal, and lemmatization
    - Data Exploration : explore the data, visualization, and feature extraction
    - Modeling : train test split, bag of word Count Vectorizer, finding TF/IDF, and using model Logistic Regression and Naive Bayes
4. Classification of Tweets to negative sentiment (0) and positive sentiment (1)
5. The image for word cloud visualization in file image

## Technologies
To execute this section you can use this technologies.
1. [JSLinux](https://bellard.org/jslinux/)
2. Jupyter Notebook
3. PuTTY
4. PySpark

## Execution
```python Twitter Social Media Analysis.py```
