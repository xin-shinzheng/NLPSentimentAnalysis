import re
import nltk
from gensim.models import Word2Vec
import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import gensim.downloader as api
import numpy as np
from sklearn.preprocessing import scale

"""
    The following are classes for various features
"""
class unigram:

    def __init__(self, dp):
        '''
        This is the constructor of the unigram class.
        :param dp: dataprocessor object
        '''
        self.dp = dp
        self.score_list = dp.df_base["Sentiment"]
        self.review_list = dp.df_base["Review"]


    def get_features(self):
        '''
        This function creates the features of unigram with bag of words.
        :param reviews: Tokenized reviews
        :param score_list: list of labels
        :return: X: features of bag of words unigram vector
                 y: labels
        '''
        bag_of_words = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False).fit_transform(
            self.review_list)
        bag_of_words = bag_of_words.toarray()

        X = bag_of_words
        y = np.array(self.score_list)

        return X, y



class pos_unigram:

    def __init__(self, dp):
        '''
        Constructor of pos_unigram which create features of
        unigram with pos bag of words.
        :param dp: dataprocessor object
        '''
        self.review_list = dp.sentences
        self.score_list = dp.df_base['Sentiment']
        self.dp=dp
    def add_pos_tag(self):
        '''
        This function cleans the reviews and add pos tags to the reviews
        :return: the tokenized reviews with pos tag
        '''
        stop_words = set(nltk.corpus.stopwords.words('english'))
        pos_l = []

        for word in self.review_list:
            clean_word = []
            word = re.sub(r'\<[^>]*\>', '', word)
            word_string = str(word)
            word_string = word_string.lower()
            tokens = nltk.word_tokenize(word_string)
            for word in tokens:
                if word not in stop_words and word not in string.punctuation:
                    clean_word.append(word)
            pos_l.append(nltk.pos_tag(clean_word))
        return pos_l


    def get_unigram_features(self, review_pos):
        '''
        This function creates the features of unigram with pos tag. It removes the
        unigram whose pos tag does not have sentiment significance.
        :param review_pos: the tokenized reviews with pos tag
        :return: X: features of reviews
                 y: labels
        '''
        pos_cloud_1 = ["JJ", "JJR", "JJS", "RB", "RBR", "RBS", "VB", "VBD",
                       "VBG", "VBN", "VBP", "VBZ", "NN", "NNS", "NNP", "NNPS", "PRP"]

        features = []
        for sent in review_pos:
            for words in sent:
                if words[1] not in pos_cloud_1:
                    sent.remove(words)
            new_review = list(zip(*sent))
            features.append(list(new_review[0]))

        bag_of_words = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False).fit_transform(
            features)
        bag_of_words = bag_of_words.toarray()

        X = bag_of_words
        y = np.array(self.score_list)
        return X, y

class pos_bigram:
    def __init__(self, dp):
        '''
        This is the constructor of the pos_bigram class which creates features of
        pos bigram bag of words.
        :param dp: dataprocessor obejct
        '''
        self.review_list = dp.sentences
        self.score_list = dp.df_base['Sentiment']
        self.dp=dp

    def clean_review(self, review_combined, dp):
        '''
        This function cleans the review.
        :param review_combined: list of reviews
        :param dp: data processor
        :return: list of strings of cleaned reviews
        '''
        review_sent = []
        for review in review_combined:
            review_sent.append(nltk.sent_tokenize(review))

        cleaned_review = []
        for review in review_sent:
            cleaned_token = []
            for sent in review:
                clean = dp.tokenize_clean(sent)
                if len(clean) != 0:
                    cleaned_token.append(clean)
            cleaned_review_string = []
            for sent in cleaned_token:
                cleaned_string = ' '.join(word for word in sent)
                cleaned_review_string.append(cleaned_string)
            cleaned_review.append(cleaned_review_string)
        return(cleaned_review)

    def sep_bigram(self, cleaned_review):
        '''
        This function separates each of the string of reviews into bigrams.
        :param cleaned_review: list of strings of cleaned reviews
        :return: the reviews separated into bigrams
        '''
        review_bigrams = []
        for review in cleaned_review:
            bigrams = [b for l in review for b in list(zip(l.split(" ")[:-1], l.split(" ")[1:]))]
            review_bigrams.append(bigrams)
        return(review_bigrams)

    def add_pos_tag(self, review_bigrams):
        '''
        This function adds pos tag fo the bigrams.
        :param review_bigrams: the reviews separated into bigrams
        :return: the reviews separated into bigrams with pos tag
        '''
        pos_review = []
        for sent in review_bigrams:
            pos_sent = []
            for bigram in sent:
                pos_bi = nltk.pos_tag(bigram)
                pos_sent.append(pos_bi)
            pos_review.append(pos_sent)
        return(pos_review)

    def remove_irrelevent(self, pos_review):
        '''
        This function removes the irrelevant bigrams which is lack of sentiment significance.
        Those are bigrams with pos tag not in the pos clouds and bigrams lack of grammatical
        structure, such as NN + ADV.
        :param pos_review: the reviews separated into bigrams with pos tag
        :return: the bigrams after removal and substitution
        '''

        # pos clouds
        pos_cloud_1 = ["JJ", "JJR", "JJS"]
        pos_cloud_2 = ["RB", "RBR", "RBS"]
        pos_cloud_3 = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
        pos_cloud_4 = ["NN", "NNS", "NNP", "NNPS", "PRP"]

        review_feature = []
        for review in pos_review:
            sent_feature = []
            for bigrams in review:
                if bigrams[0][1] in pos_cloud_1:
                    if bigrams[1][1] in pos_cloud_4:
                        string = str(bigrams[0][0] + " N")
                        sent_feature.append(string)
                    elif bigrams[1][1] in pos_cloud_2:
                        string = str(bigrams[0][0] + " RB")
                        sent_feature.append(string)
                    elif bigrams[1][1] in pos_cloud_1:
                        string = str(bigrams[0][0] + " " + bigrams[1][0])
                        sent_feature.append(string)
                if bigrams[0][1] in pos_cloud_2:
                    if bigrams[1][1] in pos_cloud_4:
                        string = str(bigrams[0][0] + " N")
                        sent_feature.append(string)
                    elif bigrams[1][1] in pos_cloud_1 or bigrams[1][1] in pos_cloud_2 or bigrams[1][1] in pos_cloud_3:
                        string = str(bigrams[0][0] + " " + bigrams[1][0])
                        sent_feature.append(string)
                if bigrams[0][1] in pos_cloud_3:
                    if bigrams[1][1] in pos_cloud_4:
                        string = str(bigrams[0][0] + " N")
                        sent_feature.append(string)
                    elif bigrams[1][1] in pos_cloud_1 or bigrams[1][1] in pos_cloud_2 or bigrams[1][1] in pos_cloud_3:
                        string = str(bigrams[0][0] + " " + bigrams[1][0])
                        sent_feature.append(string)
                if bigrams[0][1] in pos_cloud_4:
                    if bigrams[1][1] in pos_cloud_1 or bigrams[1][1] in pos_cloud_2 or bigrams[1][1] in pos_cloud_3:
                        string = str("N " + bigrams[1][0])
                        sent_feature.append(string)
            review_feature.append(sent_feature)
        return(review_feature)

    def get_feature_vector(self, review_feature, score_list):
        '''
        This function creates the feature vector using the bigram feature
        :param review_feature: the bigrams after removal and substitution
        :param score_list: labels
        :return: X: features of reviews
                 y: labels
        '''
        bag_of_words = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False).fit_transform(
            review_feature)
        bag_of_words = bag_of_words.toarray()

        X = bag_of_words
        y = np.array(score_list)

        return X, y

    def deliver_feature_vector(self):
        '''
        This function creates a bigram with pos feature.
        :return: X: features of reviews
                 y: labels
        '''

        cleaned_review = self.clean_review(self.review_list, self.dp)
        review_bigrams = self.sep_bigram(cleaned_review)
        pos_review = self.add_pos_tag(review_bigrams)
        review_feature = self.remove_irrelevent(pos_review)
        X, y = self.get_feature_vector(review_feature, self.score_list)
        return (X,y)

class lexicon:
    # Lexicon-based feature
    def lexicon_feature(self,data):
        '''
        This function extracts two lexical-based feature which are feature based
        on compound score and feature based on both negative and positive score
        :return: compound: feature based on compound score
                 neg_pos: feature based on both negative and positive score
        '''

        # Get all reviews
        review_data = data['Review'].tolist()
        review_data = [' '.join(i) for i in review_data]

        # Lecixon sentiment analysis
        analyzer = SentimentIntensityAnalyzer()

        # Get sentiment scores for each review
        scores_summary = []
        for review in review_data:
            scores = analyzer.polarity_scores(review)
            scores_summary.append(scores)

        # Extract features
        compound = [[sub['compound']] for sub in scores_summary]
        neg = [sub['neg'] for sub in scores_summary]
        pos = [sub['pos'] for sub in scores_summary]
        neg_pos = [[neg[i], pos[i]] for i in range(0, len(neg))]


        return compound, neg_pos

    def test_train(self,data, features):
        '''
        This function transforms features and labels into array X, and Y
        :param features: lexical-based feature
        :return: vectorized features and the sentiment
        '''
        # Get all sentiments
        sentiment_data = data['Sentiment'].tolist()
        # sentiment_data = [[i] for i in sentiment_data]
        X = np.array(features)
        Y = np.array(sentiment_data)

        return X, Y

class word_2_vec:
    def __init__(self,source_data):
        """
        The conscturctor for word2vec feature vectors
        :param source_data: the source data with cleaned and tokenized reviews and sentiment

        """
        self.dimension = 200
        word2vec_model_file = "amazon_food_embedding_word2vec" + str(200) + ".model"
        review_data = source_data['Review'].tolist()
        w2v_model = Word2Vec(review_data, min_count=1, size=200, workers=3, window=3, sg=1)
        w2v_model.save(word2vec_model_file)
        self.sg_w2v_model = Word2Vec.load(word2vec_model_file)
        self.X= source_data['Review'].tolist()
        self.y = source_data['Sentiment'].tolist()

    def vectorized(self,filename, df):
        """
        This function vectorizes the data with w2v and save into a new file
        :param filename:the filename to write in the vectorzied data
        :param df:The reviews needed to be vectorized
        """
        with open(filename, "w+") as file:

            index = 0
            for row in df:
                vector = (np.mean([self.sg_w2v_model[token] for token in row], axis=0)).tolist()
                if index == 0:
                    header = ",".join(str(temp) for temp in range(self.dimension))
                    file.write(header)
                    file.write("\n")
                if type(vector) is list:
                    line1 = ",".join([str(vector_element) for vector_element in vector])
                else:
                    line1 = ",".join([str[0] for i in range(self.dimension)])
                file.write(line1)
                file.write('\n')
                index = index+1

    def get_feature_vector(self):
        """
        The getter to get the vectors
        :return:vectorized reviews and sentiment
        """
        filename = "vectorized_data.csv"
        self.vectorized(filename,self.X)

        word2vec_df_train = pd.read_csv(filename).values.tolist()
        return word2vec_df_train,self.y



class glove:

    def __init__(self,source_data):
        """
        The conscturctor for word2vec feature vectors
        :param source_data: the source data with cleaned and tokenized reviews and sentiment

        """

        self.glove_twitter = api.load("glove-twitter-200")
        self.X = source_data['Review']
        self.y= source_data['Sentiment']

    def get_w2v_general(self,reviews, size, vectors):
        """
        To get the vectors for the specific reviews
        :param reviews: the reviews needed to be vectorized
        :param size: the size of dimension
        :param vectors: the pretrained vectors
        :return:the vectorized of the specific review
        """
        vec = np.zeros(size).reshape((1, size))
        count = 0.
        for word in reviews:
            try:
                vec += vectors[word].reshape((1, size))
                count += 1.
            except KeyError:
                continue
            if count != 0:
                vec /= count
            return vec

    def get_feature_vectors(self):
        """
        The getter to get the vectors
        :return:vectorized reviews and sentiment
        """

        X = scale(np.concatenate([self.get_w2v_general(review,200,self.glove_twitter) for review in self.X]))
        y = self.y
        return X, y


