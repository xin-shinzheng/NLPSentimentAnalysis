
import pandas as pd
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
class dataprocessor:
    """
        This class is for data processing
    """


    def __init__(self,filename='./Reviews_shorter.csv'):
        '''
        The constructor of data processor
        :param filename: the name of the file
        '''
        self.df_original = pd.read_csv(filename,header=0)
        self.df_original=self.df_original.dropna()
        self.sentences = []
        self.df_base = self.data_process_baseline(self.df_original)


    def data_process_baseline(self,df_original):
        '''
        The data processing baseline
        :param df_original: the dataframe from csv
        :return: cleaned and tokenized dataframe
        '''

        senti = []
        review = []
        for i, j in df_original.iterrows():
            helpfulness = 0.0
            if int(j[5]) > 0:
                helpfulness = float(int(j[4])/int(j[5]))
            if helpfulness > 0.0:
                if int(j[6]) > 3:
                    senti.append(1)

                    review.append(self.tokenize_clean(j[9]))
                    self.sentences.append(j[9])
                if int(j[6]) < 3:
                    senti.append(-1)

                    review.append(self.tokenize_clean(j[9]))
                    self.sentences.append(j[9])

        data_baseline = {"Sentiment": senti,  "Review": review}
        df_baseline = pd.DataFrame(data_baseline, columns=["Sentiment", "Review"])
        return df_baseline

    def tokenize_clean(self,sent):
        '''
        The method used for cleaning and tokenizing each reviews
        :param sent: the sentencese
        :return: the list of tokenized and cleaned tokens
        '''

        sent = re.sub(r'\<[^>]*\>', '', sent)
        tokens = word_tokenize(sent)
        tokens = [t.lower() for t in tokens]
        table = str.maketrans('','', string.punctuation)
        stripped = [t.translate(table) for t in tokens]
        words = [word for word in stripped if word.isalpha()]
        stop_words = set(stopwords.words("English"))

        words = [w for w in words if not w in stop_words]
        return words

