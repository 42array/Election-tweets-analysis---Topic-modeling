'''
Topic Modeling of US 2020 Election Tweets
'''

# ================
# INITIALIZATION
# ================
from pyspark.sql import SparkSession
from pyspark.ml.clustering import LDA
from pyspark.ml.feature import CountVectorizer, Tokenizer, StopWordsRemover, IDF
from pyspark.sql import functions as F
from pyspark.sql.functions import explode, size
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
import time
import numpy as np
from textblob import TextBlob
import pyLDAvis
from sklearn.metrics import roc_auc_score

# Create the configuration, spark context, and spark session
# spark = SparkSession.builder.master('local[*]').config('spark.executor.memory', '16g') \
#     .config('spark.driver.memory', '16g').appName('LDA Topic Modeling').getOrCreate()

spark = SparkSession.builder.master('yarn').config('spark.executor.memory', '16g')\
                                               .config('spark.driver.memory', '16g').appName('LDA Topic Modeling').getOrCreate()

spark.sparkContext.setLogLevel('ERROR')
sc = spark.sparkContext

# Define the stop words to filter output later
stop_words = ['her', 'don', 'haven', 'my', 'we', 'how', 'from', 'during', 'me', 'here', 'at', "mustn't", "you've",
              'shouldn', 'ma', 'between', 'he', 'have', 's', 'd', 'no', 'isn', 'it', 'or', 'where', 'both', 'hers',
              "you'd", "weren't", 'the', 'yourself', 't', 'after', 'with', 'themselves', 'most', 'm', 'not', "needn't",
              'our', 'into', 'than', 'being', 'in', 'below', 'its', 'once', 'while', "you're", 'she', "it's", "doesn't",
              'of', 'an', 'about', 'same', 'before', 'own', 'to', 'all', 'this', 'further', 'myself', 'down', "shan't",
              'if', 'is', 'you', 'o', 'which', 'did', 'ain', 'needn', "haven't", 'then', 'weren', 'their', 'can',
              'only', 'who', 'mightn', "should've", 'himself', 'they', 'there', 'doesn', 'out', 'ourselves', 'theirs',
              "isn't", "won't", 'am', 'over', 'your', 'aren', "mightn't", 'won', 'herself', 'y', 'wasn', 'each',
              "she's", "aren't", 'doing', 'but', 'for', 'now', 'as', 'mustn', 'against', "didn't", "couldn't", 'what',
              'had', 'again', "shouldn't", 'few', 'until', 'i', 've', 'will', 'wouldn', 'up', 'shan', 'off', 'very',
              "hasn't", "wasn't", "wouldn't", 'these', 'some', 'him', 'above', 'ours', 'yours', 'were', 'more', 'has',
              'nor', "don't", 'hasn', 'whom', 'just', 'so', 'that', 'are', 'couldn', 'those', 'because', 'on', 'such',
              'should', 'hadn', 'when', 'by', "you'll", "hadn't", "that'll", 'do', 'be', 're', 'a', 'was', 'didn',
              'does', 'll', 'why', 'itself', 'too', 'under', 'been', 'through', 'yourselves', 'his', 'them', 'other',
              'having', 'and', 'any']


# Define function to categorize tweets into 3 sentiment classes: Positive, Neutral, Negative
def sentiment_analysis(tweet_text):
    from textblob import TextBlob
    senti = TextBlob(tweet_text).sentiment[0]
    if senti == 0.0:
        return 'neutral'  # Neutral
    elif senti > 0.0:
        return 'positive'  # Positive
    elif senti < 0.0:
        return 'negative'  # Negative


# Define a function that takes in a tweet and cleans up the text (returning a cleaned version of the tweet)
# lowercases each word, removes stop words, removes short words, and removes links
def clean_tweet(tweet_text):
    cleaned_tweet = ' '.join([word.lower() for word in tweet_text.split(' ') if (word not in stop_words) \
                              and len(word) >= 4 and 'http' not in word and '&' not in word])
    return cleaned_tweet


def remove_punctuation(text):  # and extra spaces
    punc = ['.', ',', ':', ';', '{', '}', '[', ']', '-', '_', '=', '+', '(', ')', '*', '!', '$', '%', '^', '~', '`']
    for p in punc:
        text = text.replace(p, " ")
    text = " ".join(text.split())  # removing extra spaces which were being counted as the most frequent 'word'
    return text


# Define a function that takes in word ids and maps them to the actual words they represent
def id2word(topic, num_words, vocab):
    result = [vocab[topic[1][i]] for i in range(num_words)]
    return result


# Source for the two functions below: https://stackoverflow.com/questions/41819761/pyldavis-visualization-of-pyspark-generated-lda-model
# this function compiles and formats the data to match what pyLDAvis takes in as input
def format_data_to_pyldavis(tokens, cvec_model, lda_transformed, lda_model):
    counts = tokens.select((explode(tokens.clean_tokens)).alias("tokens")).groupby("tokens").count()
    wc = {i['tokens']: i['count'] for i in counts.collect()}
    wc = [wc[x] for x in cvec_model.vocabulary]

    data = {'topic_term_dists': np.array(lda_model.topicsMatrix().toArray()).T,
            'doc_topic_dists': np.array(
                [x.toArray() for x in lda_transformed.select(["topicDistribution"]).toPandas()['topicDistribution']]),
            'doc_lengths': [x[0] for x in tokens.select(size(tokens.clean_tokens)).collect()],
            'vocab': cvec_model.vocabulary,
            'term_frequency': wc}
    return data


# this function filters for bad input to pyLDAvis
def filter_bad_docs(data):
    bad = 0
    doc_topic_dists_filtrado = []
    doc_lengths_filtrado = []

    for x, y in zip(data['doc_topic_dists'], data['doc_lengths']):
        if np.sum(x) == 0:
            bad += 1
        elif np.sum(x) != 1:
            bad += 1
        elif np.isnan(x).any():
            bad += 1
        else:
            doc_topic_dists_filtrado.append(x)
            doc_lengths_filtrado.append(y)

    data['doc_topic_dists'] = doc_topic_dists_filtrado
    data['doc_lengths'] = doc_lengths_filtrado

# initialize file for writing output
out = open('/home/hadoop/averma/output_visualization.txt', 'w')

# ==================
# DATA PREPARATION
# ==================
# Load in datasets containing trump and biden tagged tweets
print("Loading in tweet data...")
print("Loading in tweet data...", file=out)
trump_sample = spark.read.csv("hdfs:///user/averma/hw3-input/archive/hashtag_donaldtrump.csv", header=True, multiLine=True)#.limit(10)
biden_sample = spark.read.csv("hdfs:///user/averma/hw3-input/archive/hashtag_joebiden.csv", header=True, multiLine=True)#.limit(10)

# Remove rows from dataframes with empty tweets
print("Removing rows with empty tweets...")
print("Removing rows with empty tweets...", file=out)
trump_sample = trump_sample.filter(trump_sample['tweet'] != '')
biden_sample = biden_sample.filter(biden_sample['tweet'] != '')

# Perform sentiment analysis on the tweet column of both dataframes (using provided function)
print("Performing sentiment analysis to create sentiment label column...")
print("Performing sentiment analysis to create sentiment label column...", file=out)
sentiment_analysis = F.udf(sentiment_analysis)
biden = biden_sample.withColumn("sentiment", sentiment_analysis(biden_sample['tweet']))
trump = trump_sample.withColumn("sentiment", sentiment_analysis(trump_sample['tweet']))

#trump = trump_sample.toPandas()
#biden = biden_sample.toPandas()
#trump['sentiment'] = trump['tweet'].map(sentiment_analysis)
#biden['sentiment'] = biden['tweet'].map(sentiment_analysis)
#trump = spark.createDataFrame(trump)
#biden = spark.createDataFrame(biden)

# trump.select('tweet').show(1, False)

# Clean up tweet text (remove stop words, lowercase all words,punctuation, etc.)
print("Cleaning tweet text...")
print("Cleaning tweet text...", file=out)
clean_tweet_udf = F.udf(lambda x: clean_tweet(x))
trump = trump.withColumn('tweet', clean_tweet_udf(trump['tweet']))
biden = biden.withColumn('tweet', clean_tweet_udf(biden['tweet']))


biden.select('tweet').show(10,truncate=False)

print("Removing punctuations")
print("Removing punctuations", file=out)
punc_udf = F.udf(lambda x: remove_punctuation(x))
biden = biden.withColumn('tweet', punc_udf(biden['tweet']))
trump = trump.withColumn('tweet', punc_udf(trump['tweet']))

# Remove rows that ended up empty after cleaning the text
trump = trump.filter(trump['tweet'] != '')
biden = biden.filter(biden['tweet'] != '')

# trump.select('tweet').show(1, False)

# print(biden.count())

# Remove rows with duplicate tweets
trump = trump.drop_duplicates(['tweet'])
biden = biden.drop_duplicates(['tweet'])

# print(biden.count())

# Tokenize the tweet text
print("Tokenizing...")
print("Tokenizing...", file=out)
tokenizer = Tokenizer(inputCol='tweet', outputCol='tokens')
biden_tokens = tokenizer.transform(biden).select('tweet', 'sentiment', 'tokens')
trump_tokens = tokenizer.transform(trump).select('tweet', 'sentiment', 'tokens')

# Remove stop words
print("Removing stop words...")
print("Removing stop words...", file=out)
remover = StopWordsRemover(inputCol='tokens', outputCol='clean_tokens')
biden_tokens = remover.transform(biden_tokens)
trump_tokens = remover.transform(trump_tokens)

## Apply count vectorizer to get the counts for each word
print("Applying Count Vectorizer...")
print("Applying Count Vectorizer...", file=out)
# for biden tweets:
vector = CountVectorizer(inputCol='clean_tokens', outputCol='count_vectors')
biden_cvec_model = vector.fit(biden_tokens.select('clean_tokens'))
biden_counts = biden_cvec_model.transform(biden_tokens.select('clean_tokens'))  # clean_tokens, count_vectors

# for trump tweets:
vector = CountVectorizer(inputCol='clean_tokens', outputCol='count_vectors')
trump_cvec_model = vector.fit(trump_tokens.select('clean_tokens'))
trump_counts = trump_cvec_model.transform(trump_tokens.select('clean_tokens'))  # clean_tokens, count_vectors

## Use the counts to apply TF-IDF vectorization
print("Using counts for TF-IDF vectorization...")
print("Using counts for TF-IDF vectorization...", file=out)
# for biden tweets:
tfidf = IDF(inputCol='count_vectors', outputCol='features')
biden_tfidf_model = tfidf.fit(biden_counts)
biden_tfidf = biden_tfidf_model.transform(biden_counts)  # clean_tokens, count_vectors, features

# for trump tweets:
tfidf = IDF(inputCol='count_vectors', outputCol='features')
trump_tfidf_model = tfidf.fit(trump_counts)
trump_tfidf = trump_tfidf_model.transform(trump_counts)  # clean_tokens, count_vectors, features

biden_tfidf.show()
# trump_tfidf.show()


# ===================
# LDA TOPIC MODELING
# ===================
number_of_topics = [3,5,7,10,15]
for num_topics in number_of_topics:
    # Biden Tweets LDA Model
    print("Fitting Biden tweets LDA model for {} topics...".format(num_topics))
    print("Fitting Biden tweets LDA modelfor {} topics...".format(num_topics), file=out)
    #num_topics = 5
    max_iter = 20
    lda = LDA(seed=1, optimizer="em", k=num_topics, maxIter=max_iter)
    biden_lda_model = lda.fit(biden_tfidf.select('features'))
    biden_lda_transformed = biden_lda_model.transform(biden_tfidf.select('features'))

# biden_lda_transformed.show()

# Get topics and words
    print("Getting topics from Biden LDA model...")
    print("Getting topics from Biden LDA model...", file=out)
    biden_topics = biden_lda_model.topicsMatrix()
    biden_vocab = biden_cvec_model.vocabulary

    num_words = 30  # specify number of words per topic
    topic_word_ids = biden_lda_model.describeTopics(maxTermsPerTopic=num_words).rdd.map(tuple)  # get ids of words in topics
    biden_topics_final = topic_word_ids.map(
        lambda x: id2word(x, num_words, biden_vocab)).collect()  # get the list of words for each topic

    for i in range(len(biden_topics_final)):
        print("Topic " + str(i + 1) + ":")
        print(biden_topics_final[i])

    # Data Visualization
    print("Visualizing Biden tweet topics using pyLDAvis...")
    #timeStamp = str(int(time()))
    data = format_data_to_pyldavis(biden_tokens, biden_cvec_model, biden_lda_transformed, biden_lda_model)
    filter_bad_docs(data)
    py_lda_prepared_data = pyLDAvis.prepare(**data)
    pyLDAvis.save_html(py_lda_prepared_data, '/home/hadoop/averma/biden_data-viz-' + str(num_topics) + '.html')


# Trump Tweets LDA Model
    print("Fitting Trump tweets LDA model for {} topics...".format(num_topics))
    print("Fitting Trump tweets LDA modelfor {} topics...".format(num_topics), file=out)
    #num_topics = 5
    max_iter = 20
    lda = LDA(seed=1, optimizer="em", k=num_topics, maxIter=max_iter)
    trump_lda_model = lda.fit(trump_tfidf.select('features'))
    trump_lda_transformed = trump_lda_model.transform(trump_tfidf.select('features'))

    # trump_lda_transformed.show()

    # Get topics and words
    print("Getting topics from Trump LDA model...")
    print("Getting topics from Trump LDA model...", file=out)
    trump_topics = trump_lda_model.topicsMatrix()
    trump_vocab = trump_cvec_model.vocabulary

    num_words = 30  # specify number of words per topic
    topic_word_ids = trump_lda_model.describeTopics(maxTermsPerTopic=num_words).rdd.map(tuple)  # get ids of words in topics
    trump_topics_final = topic_word_ids.map(
        lambda x: id2word(x, num_words, trump_vocab)).collect()  # get the list of words for each topic

    for topic in range(len(trump_topics_final)):
        print("Topic " + str(topic) + ":")
        print(trump_topics_final[topic])

    # Data Visualization
    print("Visualizing Trump tweet topics using pyLDAvis...")
    #timeStamp = str(int(time()))
    data = format_data_to_pyldavis(trump_tokens, trump_cvec_model, trump_lda_transformed, trump_lda_model)
    filter_bad_docs(data)
    py_lda_prepared_data = pyLDAvis.prepare(**data)
    pyLDAvis.save_html(py_lda_prepared_data, '/home/hadoop/averma/trump_data-viz-' + str(num_topics) + '.html')


# =======================================================================
# SENTIMENT CLASSIFICATION USING LOGISTIC REGRESSION AND TF-IDF FEATURES
# =======================================================================
# Perform joins to compile all the data necessary to run logistic regression
# End goal is to have the following in one df: features, topicDistribution, sentiment
print("Compiling data for logistic regression...")
print("Compiling data for logistic regression...", file=out)
biden_join1 = biden_tokens.select('tweet', 'sentiment', 'clean_tokens').join(
    biden_tfidf.select('clean_tokens', 'features'),
    on='clean_tokens', how='outer')  # tweet, sentiment, clean_tokens, features

biden_final = biden_join1.join(biden_lda_transformed, on='features',
                               how='outer')  # tweet, sentiment, clean_tokens, features, topicDistribution

trump_join1 = trump_tokens.select('tweet', 'sentiment', 'clean_tokens').join(
    trump_tfidf.select('clean_tokens', 'features'),
    on='clean_tokens', how='outer')  # tweet, sentiment, clean_tokens, features

trump_final = trump_join1.join(trump_lda_transformed, on='features',
                               how='outer')  # tweet, sentiment, clean_tokens, features, topicDistribution


# Initialize helper functions for logistic regression:
def index_labels(df):  # index sentiment labels
    labelIndexer = StringIndexer().setInputCol("sentiment").setOutputCol("sentiment_index")  # index sentiment strings
    df = labelIndexer.fit(df).transform(df)
    return df


def topic_dist_df(df):  # Include topic distributions, make feature vectors and index sentiment labels
    assembler = VectorAssembler(inputCols=['features', 'topicDistribution'], outputCol="vector_features")
    df_vector = assembler.transform(df)  # vectors
    df_vector = df_vector.select("vector_features", "sentiment")
    labelIndexer = StringIndexer().setInputCol("sentiment").setOutputCol("sentiment_index")  # index sentiment strings
    df_vector1 = labelIndexer.fit(df_vector).transform(df_vector)
    return df_vector1


# def logistic_regression_fn(lr, train_df, val_df):  # LR with no cross val
#     lrModel = lr.fit(train_df)
#     predictions = lrModel.transform(val_df)
#     evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",
#                                                   labelCol='sentiment_index')
#     score = evaluator.evaluate(predictions)
#     return score

# runs cross validation to get the best logistic regression model and returns it
def logistic_regression_CV(lr, train_df, reg_params, max_iters, elasticnet_params, numFolds):  # LR with cross val
    paramGrid = (ParamGridBuilder()
                 .addGrid(lr.regParam, reg_params)
                 .addGrid(lr.maxIter, max_iters)
                 .addGrid(lr.elasticNetParam, elasticnet_params)
                 .build())

    evaluator = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='sentiment_index')

    crossVal = CrossValidator(estimator=lr, evaluator=evaluator, numFolds=numFolds, estimatorParamMaps=paramGrid)
    cv_model = crossVal.fit(train_df)

    return cv_model


print("Performing logistic regression using the TF-IDF representation")
print("Performing logistic regression using the TF-IDF representation", file=out)

# Biden: logistic regression (TF-IDF only) w/ cross val
reg_params = [0.01]       #[0.01,0.05]
max_iters = [10]          #[10,20]
elasticnet_params = [0.5] #[0.1,0.5,0.9]
numFolds = 10

lr = LogisticRegression(featuresCol='features', labelCol='sentiment_index')
split_ratio = [0.8, 0.2]
biden_feat = biden_final.select('features', 'sentiment')
biden_feat = index_labels(biden_feat)
biden_train, biden_val = biden_feat.randomSplit(split_ratio)

biden_best_model = logistic_regression_CV(lr, biden_train, reg_params, max_iters, elasticnet_params, numFolds)
fullPredictions = biden_best_model.transform(biden_val)
probabilities = fullPredictions.select('probability').rdd.map(lambda x: x[0]).collect()
labels = fullPredictions.select('sentiment_index').rdd.map(lambda x: x[0]).collect()

biden_ROC = roc_auc_score(labels, probabilities, multi_class="ovo", average="macro", labels=np.arange(3))

print("Biden logistic regression (TF-IDF) AUC-ROC score: ", biden_ROC)
print("Biden logistic regression (TF-IDF) AUC-ROC score: ", biden_ROC, file=out)


# Trump: logistic regression (TF-IDF only) w/ cross val
reg_params = [0.01]       #[0.01,0.05]
max_iters = [10]          #[10,20]
elasticnet_params = [0.5] #[0.1,0.5,0.9]
numFolds = 10

lr = LogisticRegression(featuresCol='features', labelCol='sentiment_index')
split_ratio = [0.8, 0.2]
trump_feat = trump_final.select('features', 'sentiment')
trump_feat = index_labels(trump_feat)
trump_train, trump_val = trump_feat.randomSplit(split_ratio)

trump_best_model = logistic_regression_CV(lr, trump_train, reg_params, max_iters, elasticnet_params, numFolds)
fullPredictions = trump_best_model.transform(trump_val)
probabilities = fullPredictions.select('probability').rdd.map(lambda x: x[0]).collect()
labels = fullPredictions.select('sentiment_index').rdd.map(lambda x: x[0]).collect()

trump_ROC = roc_auc_score(labels, probabilities, multi_class="ovo", average="macro", labels=np.arange(3))

print("Trump logistic regression (TF-IDF) AUC-ROC score: ", trump_ROC)
print("Trump logistic regression (TF-IDF) AUC-ROC score: ", trump_ROC, file=out)


# =============================================================================================
# SENTIMENT CLASSIFICATION USING LOGISTIC REGRESSION, TF-IDF FEATURES, AND TOPIC DISTRIBUTIONS
# =============================================================================================
print("Performing logistic regression using the TF-IDF representation and Topic distributions")
print("Performing logistic regression using the TF-IDF representation and Topic distributions", file=out)
# Biden: logistic regression (TF-IDF and Topics) w/ cross val
reg_params = [0.01]       #[0.01,0.05]
max_iters = [10]          #[10,20]
elasticnet_params = [0.5] #[0.1,0.5,0.9]
numFolds = 10

lr_TD = LogisticRegression(featuresCol='vector_features', labelCol='sentiment_index')
split_ratio = [0.8, 0.2]
biden_TD = topic_dist_df(biden_final)  # add topic distributions and get feature vectors
biden_train_TD, biden_val_TD = biden_TD.randomSplit(split_ratio)

biden_best_model_TD = logistic_regression_CV(lr_TD, biden_train_TD, reg_params, max_iters, elasticnet_params, numFolds)
fullPredictions = biden_best_model_TD.transform(biden_val_TD)
probabilities = fullPredictions.select('probability').rdd.map(lambda x: x[0]).collect()
labels = fullPredictions.select('sentiment_index').rdd.map(lambda x: x[0]).collect()

biden_ROC_TD = roc_auc_score(labels, probabilities, multi_class="ovo", average="macro", labels=np.arange(3))

print("Biden logistic regression (TF-IDF + Topics) AUC-ROC score: ", biden_ROC_TD)
print("Biden logistic regression (TF-IDF + Topics) AUC-ROC score: ", biden_ROC_TD, file=out)


# Trump: logistic regression (TF-IDF and Topics) w/ cross val
reg_params = [0.01]       #[0.01,0.05]
max_iters = [10]          #[10,20]
elasticnet_params = [0.5] #[0.1,0.5,0.9]
numFolds = 10

lr_TD = LogisticRegression(featuresCol='vector_features', labelCol='sentiment_index')
split_ratio = [0.8, 0.2]
trump_TD = topic_dist_df(trump_final)  # add topic distributions and get feature vectors
trump_train_TD, trump_val_TD = trump_TD.randomSplit(split_ratio)

trump_best_model_TD = logistic_regression_CV(lr_TD, trump_val_TD, reg_params, max_iters, elasticnet_params, numFolds)
fullPredictions = trump_best_model_TD.transform(trump_val_TD)
probabilities = fullPredictions.select('probability').rdd.map(lambda x: x[0]).collect()
labels = fullPredictions.select('sentiment_index').rdd.map(lambda x: x[0]).collect()

trump_ROC_TD = roc_auc_score(labels, probabilities, multi_class="ovo", average="macro", labels=np.arange(3))

print("Trump logistic regression (TF-IDF + Topics) AUC-ROC score: ", trump_ROC_TD)
print("Trump logistic regression (TF-IDF + Topics) AUC-ROC score: ", trump_ROC_TD, file=out)


