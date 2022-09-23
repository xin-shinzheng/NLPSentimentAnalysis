import dataprocessor as dp
import classifiers as cla
import features as ft
import time
def print_baseline_nb(X,y):
    """
    Baseline model only run Multinomial Naive bayes
    :param X: The tokenized reviews
    :param y: The lables
    """
    classifier = cla.classifiers(X,y,1)
    classifier.get_classifiers(1,0,0)

def print_three_classifiers_results(X,y,svm,multinomial):
    """
    Method to run and print the evaluation metrics of three classifiers seperately
    :param X: The tokenized reviews
    :param y: The lables
    :param svm: Whether uses SVM
    :param multinominal: Whether uses Multinomial

    """
    if multinomial:
        classifier = cla.classifiers(X,y,1)
    else:
        classifier = cla.classifiers(X,y,0)
    #nb
    classifier.get_classifiers(1, 0, 0)
    #max entropy
    classifier.get_classifiers(0, 1, 0)
    #svm
    if svm:
        classifier.get_classifiers(0, 0, 1)


if __name__ == '__main__':
    """
        This is the main method 
    """

    df = dp.dataprocessor('./Reviews_new.csv')
    source_data = df.df_base
    #baseline
    baseline = ft.unigram(df)
    X, y =baseline.get_features()
    print("the cross validation scores for baseline model ")
    print_baseline_nb(X,y)

    #unigram+ pos  features
    pos_uni = ft.pos_unigram(df)
    reviews_pos = pos_uni.add_pos_tag()
    X,y = pos_uni.get_unigram_features(reviews_pos)
    print("the cross validation scores for unigram + POS: ")
    print_three_classifiers_results(X,y,1,1)
    #bigram features
    pos_bi = ft.pos_bigram(df)
    X,y = pos_bi.deliver_feature_vector()
    print("the cross validation scores for bigram + POS: ")
    print_three_classifiers_results(X,y,0,1)
    
    #lexical compound features
    lexi = ft.lexicon()
    feature_compound, feature_neg_pos = lexi.lexicon_feature(source_data)
    X, y = lexi.test_train(source_data, feature_compound)
    print("the cross validation scores for lexical compound: ")
    print_three_classifiers_results(X,y,1,0)

    
    #lexical neg+pos features
    X, y = lexi.test_train(source_data, feature_neg_pos)
    print("the cross validation scores for lexical neg pos: ")
    print_three_classifiers_results(X,y,1,0)
    
    #word2vec
    w2v = ft.word_2_vec(source_data)
    X,y = w2v.get_feature_vector()
    print("the cross validation scores for word2vec: ")
    print_three_classifiers_results(X,y,1,0)

    #glove
    print("start creating glv features")
    start_time = time.time()
    glv = ft.glove(source_data)
    X, y = glv.get_feature_vectors()
    print("createing glv feature vectors takes " + str(time.time()-start_time))
    print("the cross validation scores for GLoVe: ")
    print_three_classifiers_results(X,y,1,0)

