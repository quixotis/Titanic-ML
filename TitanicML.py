#importing libraries
import pandas as pd
import numpy as np
from sklearn import linear_model, svm
from sklearn.ensemble import RandomForestClassifier, VotingClassifier



##~Handling train data~

#load training data into dataframe
train = pd.read_csv("/Users/Quixotis/Desktop/train.csv")


#converting gender categorical data into numbers
train["Sex"][train["Sex"]=="male"]=0
train["Sex"][train["Sex"]=="female"]=1

#converting embarkation categorical data into numbers
train["Embarked"][train["Embarked"]=="S"]=1
train["Embarked"][train["Embarked"]=="C"]=2
train["Embarked"][train["Embarked"]=="Q"]=3

#filling in missing age data with mean age
average_age = train["Age"].mean()
train["Age"] = train["Age"].fillna(average_age)

#combining sibsp and parch into single feature and removing irrelevant features from dataframe
train["Family"] = train["SibSp"]+train["Parch"]
train = train[["Survived","Pclass","Sex","Age","Family","Fare","Embarked"]]

#fill in remaining missing data with zeros
train = train.fillna(0)

#for loop that generates a "child" feature based on being younger than 18
train["Child"] = 0
for x in range(0, 891):
 	if train["Age"][x]>=18:
 		train["Child"][x]=0
 	elif train["Age"][x]<18:
 		train["Child"][x]=1

#sepearting "survived" from dataframe and creating two separate arrays for logistic regression
target = train["Survived"]
target_data = target.values
train_data = train.drop("Survived", 1)
train_data = train_data.values



##~Handling test data~

#loading test data and creating test submission dataframe
test = pd.read_csv("/Users/Quixotis/Desktop/test.csv")
test_submission = pd.read_csv("/Users/Quixotis/Desktop/test.csv")
test_submission = test_submission.drop(["Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"], 1)
testlog_submission = test_submission


#formatting test data
test["Sex"][test["Sex"]=="male"]=0
test["Sex"][test["Sex"]=="female"]=1
test["Embarked"][test["Embarked"]=="S"]=1
test["Embarked"][test["Embarked"]=="C"]=2
test["Embarked"][test["Embarked"]=="Q"]=3

#filling in missing age data with mean age or other values imputed with 0
average_age = test["Age"].mean()
test["Age"] = test["Age"].fillna(average_age)
test = test.fillna(0)

#for loop that generates a "child" feature based on being younger than 18 for the test data
test["Child"] = 0
for x in range(0, 418):
 	if test["Age"][x]>=18:
 		test["Child"][x]=0
 	elif test["Age"][x]<18:
 		test["Child"][x]=1

#creates "family" feature and then preps dataframe to become test data
test["Family"] = test["SibSp"]+test["Parch"]
test = test[["Pclass","Sex","Age","Family","Fare","Embarked", "Child"]]
test_data = test.values



##~Implementing classifiers~

#initializing logistic regression and displaying accuracy on training set
log_regression = linear_model.LogisticRegression()
log_regression = log_regression.fit(train_data, target_data)
log_score = log_regression.score(train_data, target_data)
print(log_score)

#initializing support vector classification and displaying accuracy on training set
svm_analysis = svm.SVC(probability=True)
svm_analysis = svm_analysis.fit(train_data, target_data)
svm_score = svm_analysis.score(train_data, target_data)
print(svm_score)

#initializing random forest classification and displaying accuracy on training set
random_forest = RandomForestClassifier(n_estimators=100)
random_forest = random_forest.fit(train_data, target_data)
forest_score = random_forest.score(train_data, target_data)
print(forest_score)

#initializing voting classifier and dispalying accuracy on training set
vote_classifier = VotingClassifier(estimators=[('lr', log_regression), ('rf', random_forest), ('svc', svm_analysis)], voting='soft') #implemented as soft to take advantage of probabilities instead of binary values
vote_classifier = vote_classifier.fit(train_data, target_data)
vote_score = vote_classifier.score(train_data, target_data)
print(vote_score)

#creating prediction dataframe
vote_prediction = vote_classifier.predict(test_data)
vote_prediction = pd.DataFrame(vote_prediction)
vote_prediction.columns = ["Survived"]

#finalizing the test submission file and exporting it
test_submission["Survived"] = vote_prediction["Survived"]
test_submission = test_submission.set_index("PassengerId")
test_submission.to_csv("/Users/Quixotis/Desktop/test_submission.csv")












