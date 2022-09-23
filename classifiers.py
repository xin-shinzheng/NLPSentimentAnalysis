import warnings

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
import time

class classifiers:
    '''
        This is a class containing three classifiers, Naive Bayes,
        Maximum Entropy, and SVM.
    '''

    def __init__(self,X,y,multinominal):
        '''
        This is the constructor of the class. Member variables are
        X and y.
        :param X: The features of the reviews
        :param y: Labels of the reviews
        :param multinominal: Naive Bayes classifier
        '''
        self.X = X
        self.y=y
        self.multinominal = multinominal

    def get_classifiers(self,nb, maxentro, svm):
        '''
        This method train the classifier, naive bayes, maximum entropy, and SVM
         with 5-fold cross validation using X and y.
        :param nb: flag of naive bayes
        :param maxentro: flag of maximum entropy
        :param svm: flag of svm
        :return: the lis of strings indicating the scores of accuracy, f1, precision and
                 recall of different classifiers
        '''
        with warnings.catch_warnings():
            # ignore all caught warnings
            warnings.filterwarnings("ignore")
        # execute code that will generate warnings
            if nb :

                start_time = time.time()
                if self.multinominal:
                    print("start Multinomial Naive Bayes")
                    clf = MultinomialNB()
                else:
                    print("start Gaussian Naive Bayes")
                    clf = GaussianNB()
            if maxentro:
                print("start Maximum Entropy ")
                start_time = time.time()
                clf = linear_model.LogisticRegression(penalty='l2', C=1.0, max_iter=1000)
            if svm:
                print("start Support Vector Machine")
                start_time = time.time()
                clf = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', probability=True, class_weight='balanced'),
                                                        max_samples=1.0 / 10, n_estimators=10))

            scoring = ['accuracy','f1_macro','precision_macro', 'recall_macro']
            scores = cross_validate(clf, self.X, self.y, scoring=scoring)

            accuracy = scores['test_accuracy']
            f1 = scores['test_f1_macro']
            recall = scores['test_recall_macro']
            precision = scores['test_precision_macro']
            accuracy_string = "Accuracy with 5-folds: %0.2f (+/- %0.2f)" % (accuracy.mean(), accuracy.std() * 2)
            f1_string = "F1 macro with 5-folds: %0.2f (+/- %0.2f)" % (f1.mean(), f1.std() * 2)
            precision_string = "Precision macro with 5-folds: %0.2f (+/- %0.2f)" % (precision.mean(), precision.std() * 2)
            recall_string = "Recall macro with 5-folds: %0.2f (+/- %0.2f)" % (recall.mean(), recall.std() * 2)
            string_holder = [accuracy_string,f1_string,precision_string,recall_string]
            for string in string_holder:
                print(string)
            print("running model takes " + str(time.time()-start_time) + " sceonds")
            print()
        return string_holder

