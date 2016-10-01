import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random
random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
import math
#nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    
    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg



def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))
    
    # Determine a list of words that will be used as features. 
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    # YOUR CODE HERE

    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    # YOUR CODE HERE

    mydict={}
    wordlist=[]
    train_pos_strip = []
    train_neg_strip = []
    test_pos_strip = []
    test_neg_strip = []

    for text in train_pos:
        for word in text:
            if word not in stopwords and word not in wordlist:
                if word not in mydict:
                    mydict[word] = [1,0]
                else:
                    mydict[word][0] += 1
                wordlist.append(word)
        train_pos_strip.append(wordlist)
        wordlist = []

    for text in train_neg:
        for word in text:
            if word not in stopwords and word not in wordlist:
                if word not in mydict:
                    mydict[word] = [0,1]
                else:
                    mydict[word][1] += 1
                wordlist.append(word)
        train_neg_strip.append(wordlist)
        wordlist = []


    for text in test_pos:
        for word in text:
            if word not in stopwords and word not in wordlist:
                wordlist.append(word)
        test_pos_strip.append(wordlist)
        wordlist = []
    
    for text in test_neg:
        for word in text:
            if word not in stopwords and word not in wordlist:
                wordlist.append(word)
        test_neg_strip.append(wordlist)
        wordlist = []

    #lengthPos = math.ceil(len(train_pos)+len(test_pos) * 0.01)
    #lengthNeg = math.ceil(len(train_neg)+len(test_neg) * 0.01)
    lengthPos = math.ceil(len(train_pos) * 0.01)
    lengthNeg = math.ceil(len(train_neg) * 0.01)
    
    listWord = []
    for key in mydict:
        if (mydict[key][0]>=lengthPos or mydict[key][1]>=lengthNeg) and (mydict[key][0]>=2*mydict[key][1] or mydict[key][0]<=2*mydict[key][1]):
            listWord.append(key)

    train_pos_vec = []
    train_neg_vec = []
    test_pos_vec = []
    test_neg_vec = []

    for text in train_pos_strip:
        featureVec = []
        for word in listWord:
            if word in text:
                featureVec.append(1)
            else:
                featureVec.append(0)
        train_pos_vec.append(featureVec)

    for text in train_neg_strip:
        featureVec = []
        for word in listWord:
            if word in text:
                featureVec.append(1)
            else:
                featureVec.append(0)
        train_neg_vec.append(featureVec)

    for text in test_pos_strip:
        featureVec = []
        for word in listWord:
            if word in text:
                featureVec.append(1)
            else:
                featureVec.append(0)
        test_pos_vec.append(featureVec)

    for text in test_neg_strip:
        featureVec = []
        for word in listWord:
            if word in text:
                featureVec.append(1)
            else:
                featureVec.append(0)
        test_neg_vec.append(featureVec)

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec

def labelize(data, label_type):
    labelized = []
    for i,v in enumerate(data):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(words = v, tags = [label]))
    return labelized

'''
def extractFeatureVec(model,length,label_type):
    vec = []
    for i in range(length):
        label = '%s_%s'%(label_type,i)
        vec.append(list(model.docvecs[label]))
    return vec
'''

def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    # YOUR CODE HERE

    labeled_train_pos = labelize(train_pos,'train_pos')
    labeled_train_neg = labelize(train_neg,'train_neg')
    labeled_test_pos = labelize(test_pos,'test_pos')
    labeled_test_neg = labelize(test_neg,'test_neg')

    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run 
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)

    # Use the docvecs function to extract the feature vectors for the training and test data
    # YOUR CODE HERE
    #train_pos_vec = extractFeatureVec(lr_modelel,len(train_pos),'train_pos')
    #train_neg_vec = extractFeatureVec(model,len(train_neg),'train_neg')
    #test_pos_vec = extractFeatureVec(model,len(test_pos),'test_pos')
    #test_neg_vec = extractFeatureVec(model,len(test_neg),'test_neg')

    train_pos_vec=[]
    train_neg_vec=[]
    test_pos_vec=[]
    test_neg_vec=[]
    for label in model.docvecs.doctags.keys():
        if 'train_pos' in label:
            train_pos_vec.append(model.docvecs[label])
        elif 'train_neg' in label:
            train_neg_vec.append(model.docvecs[label])
        elif 'test_pos' in label:
            test_pos_vec.append(model.docvecs[label])
        elif 'test_neg' in label:
            test_neg_vec.append(model.docvecs[label])

    #print test_neg_vec
    
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    traindata = train_pos_vec + train_neg_vec

    nb_model = sklearn.naive_bayes.BernoulliNB(alpha=1.0,binarize=None)
    nb_model.fit(traindata,Y)
    lr_model = sklearn.linear_model.LogisticRegression()
    lr_model.fit(traindata, Y)
    
    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    traindata = train_pos_vec + train_neg_vec

    nb_model = sklearn.naive_bayes.GaussianNB()
    nb_model.fit(traindata,Y)
    lr_model = sklearn.linear_model.LogisticRegression()
    lr_model.fit(traindata,Y)
    
    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    # YOUR CODE HERE
    test_pos_label = list(model.predict(test_pos_vec))
    tp = test_pos_label.count('pos')
    fn = len(test_pos_label) - tp

    test_neg_label = list(model.predict(test_neg_vec))
    fp = test_neg_label.count('pos')
    tn = len(test_neg_label) - fp
    accuracy = float(tp+tn) / (tp+tn+fn+fp)
    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)



if __name__ == "__main__":
    main()
