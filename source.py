from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

clf_decision=tree.DecisionTreeClassifier()
# clf_random=tree.RandomForestClassifier()
clf_ada=AdaBoostClassifier()
clf_KNeighbor=KNeighborsClassifier()

#[Height,weight,shoe_size]

X=[[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],[190, 90, 47], 
[175, 64, 39],[177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y=['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female','female', 'male', 
'male']

clf_KNeighbor=clf_KNeighbor.fit(X,Y)
clf_ada=clf_ada.fit(X,Y)
# clf_random=clf_random.fit(X,Y)
clf_decision=clf_decision.fit(X,Y)

Prediction_KNeighbor=clf_KNeighbor.predict([[190, 70, 43]])
Prediction_ada=clf_ada.predict([[190,70,43]])
# Prediction_random=clf_random.predict([[190,70,43]])
Prediction_decision=clf_decision.predict([[190,70,43]])

print(Prediction_decision)
# print(Prediction_random)
print(Prediction_ada)
print(Prediction_KNeighbor)
