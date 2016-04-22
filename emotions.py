__author__ = 'lukas.bitter'
# !/usr/bin/env python
# -*- coding: utf-8 -*-

import sklearn
import sklearn.datasets
import os
import random
from shutil import copyfile

class FileTool:
    def __init__(self, wordsToIgnore, dataFolder):
        self.wordsToIgnore = wordsToIgnore
        self.dataFolder = dataFolder
        self.taggedFolder = dataFolder + '\\tagged'
        self.runFolderName = dataFolder + '\\runTP'
        self.trainFolder = self.runFolderName + '\\train'
        self.testFolder = self.runFolderName + '\\test'
        self.categories = []
        self.listStopWords = []
        self.stopWordsFile = 'frenchST.txt'
        self.significantWords = ['NOM', 'VER' ,'ADV', 'ADJ']

        self.stopWords()
        self.initFileSystem()
        self.train()

    def extractUnsignificantWords(self, fileName):
        try:
            file = open(fileName, mode='r', encoding='utf-8')
        except:
            raise Exception('Nom de fichier: ', file, ' INVALIDE\n')

        newFile = []
        for line in file:
            words = line.split('\t')
            try:
                if words[1][:3] in self.significantWords:
                    newFile.append(words[2])
            except:
                print(file, ': ', words)

        file.close()
        fileW = open(fileName, mode='w', encoding='utf-8')
        fileW.writelines(newFile)
        fileW.close()

    def initFileSystem(self):
        # create categories]
        for cat in os.listdir(self.taggedFolder):
            self.categories.append(cat)
            taggedCatFolder = self.taggedFolder + '\\' + cat

            # manage train files
            fRunFolderTrain = self.trainFolder + '\\' + cat
            fileList = os.listdir(taggedCatFolder)
            if not os.path.exists(fRunFolderTrain):
                os.makedirs(fRunFolderTrain)
                for f in fileList:
                    copyfile(taggedCatFolder + '\\' + f, fRunFolderTrain + '\\' + f)
                    self.extractUnsignificantWords(fRunFolderTrain + '\\' + f)

            # manage test files
            fRunFolderTest = self.runFolderName + '\\test' + '\\' + cat
            if not os.path.exists(fRunFolderTest):
                os.makedirs(fRunFolderTest)
                testList = random.sample(fileList, 200)
                for f in testList:
                    os.rename(fRunFolderTrain + '\\' + f, fRunFolderTest + '\\' + f)

    def train(self):
        movies_train=sklearn.datasets.load_files(self.trainFolder,
                                         description=None,
                                         categories=self.categories,
                                         load_content=True,
                                         shuffle=True,
                                         encoding='latin-1',
                                         decode_error='strict',
                                         random_state=42)

        #print('movies_train.target_names: ',  movies_train.target_names)
        # ['neg', 'pos'] --> folders

        #print('len(movies_train.data): ', len(movies_train.data))
        # 1600 --> distinct documents

        from sklearn.feature_extraction.text import CountVectorizer
        count_vect = CountVectorizer(stop_words = self.listStopWords)
        X_train_counts = count_vect.fit_transform(movies_train.data)

        #print('X_train_counts.shape: ', X_train_counts.shape)
        # (1600, 12636) --> for 1600 documents, there are 12636 different words

        from sklearn.feature_extraction.text import TfidfTransformer
        tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
        X_train_tf = tf_transformer.transform(X_train_counts)
        #print('X_train_tf.shape: ', X_train_tf.shape)

        from sklearn.naive_bayes import MultinomialNB
        clf = MultinomialNB().fit(X_train_tf, movies_train.target)

        from sklearn.pipeline import Pipeline
        text_clf = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('clf', MultinomialNB()),
        ])

        text_clf = text_clf.fit(movies_train.data, movies_train.target)

        import numpy as np
        movies_test=sklearn.datasets.load_files(self.testFolder,
                                         description=None,
                                         categories=self.categories,
                                         load_content=True,
                                         shuffle=True,
                                         encoding='latin-1',
                                         decode_error='strict',
                                         random_state=42)
        docs_test = movies_test.data
        predicted = text_clf.predict(docs_test)
        print(np.mean(predicted == movies_test.target))

        from sklearn.linear_model import SGDClassifier
        text_clf = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                                   alpha=1e-3, n_iter=5, random_state=42)),
        ])
        _ = text_clf.fit(movies_train.data, movies_train.target)
        predicted = text_clf.predict(docs_test)
        print(np.mean(predicted == movies_test.target))

        from sklearn import metrics
        print(metrics.classification_report(movies_test.target, predicted,
        target_names=movies_test.target_names))
        print(metrics.confusion_matrix(movies_test.target, predicted))

        from sklearn.grid_search import GridSearchCV
        parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                      'tfidf__use_idf': (True, False),
                      'clf__alpha': (1e-2, 1e-3),
        }

        gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
        gs_clf = gs_clf.fit(movies_train.data, movies_train.target)

        best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, best_parameters[param_name]))

        print(score)

    def stopWords(self):
        try:
            file = open(self.stopWordsFile)
        except:
            raise Exception('Nom de fichier: ', self.stopWordsFile, ' INVALIDE\n')

        for ligne in file:
            self.listStopWords.append(ligne)

if __name__ == '__main__':
    wordsToIgnore = 'frenchST.txt' #input('Saisir un nom de fichier contenant les mots à ignorer: ')
    dataFolder = '..\\data' #input('répertoire contenant les categories de fichiers à analyser')
    myFile = FileTool(wordsToIgnore, dataFolder)
