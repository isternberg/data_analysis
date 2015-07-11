import data_preparation as dp

# Train with decision tree algorithm
from sklearn import tree, cross_validation
decision_tree = tree.DecisionTreeClassifier()
#cv is number of folds
scores_tree = cross_validation.cross_val_score(decision_tree, dp.x_train, dp.y_train, cv=5)
print "Cross Validation with decision tree:"
print scores_tree


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
score_bayes = cross_validation.cross_val_score(clf, dp.x_train, dp.y_train, cv=5)
print "Cross Validation with naive bayes:"
print score_bayes


# train
decision_tree = decision_tree.fit(dp.x_train, dp.y_train)
# predict - TODO the same with gnb and clf
prediction = decision_tree.predict(dp.x_test)
# test
from sklearn.metrics import accuracy_score
print "Accuracy of prediction with decision tree:"
print accuracy_score(dp.y_test, prediction)  # 0.2

# train
naive_bayes = clf.fit(dp.x_train, dp.y_train)
# predict - TODO the same with gnb and clf
prediction = clf.predict(dp.x_test)
print "Accuracy of prediction with naive bayes:"
print accuracy_score(dp.y_test, prediction)  # 0.2