from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

app= Flask(__name__)


df = pd.read_csv('data.csv')
# Rename the columns
new_col = ['age','address','schooling','stud_hr','employed','h_disab',
           'ment_cond','social_hr','fit_hr','wind','dry_mouth',
           'positive','breath_diff','initiate','tremb','worry','look_fwd',
           'down','enthus','life_mean','scared','outcome']
           
df.columns = new_col

# We'll rename and replace

df['outcome'] = df['outcome'].str.replace('high signs of depression ','High signs of depression ')

# Dropping schooling and address columns because they have only 1  values each

df.drop(['schooling','address'],axis=1, inplace=True)

# Split dataset
X, y = df.iloc[:, :-1], df.iloc[:, -1]

# Create train and test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)



# Data Scaling

# First we need to know which columns are binary, nominal and numerical
def get_columns_by_category():
    categorical_mask = X.select_dtypes(
        include=['object']).apply(pd.Series.nunique) == 2
    numerical_mask = X.select_dtypes(
        include=['int64', 'float64']).apply(pd.Series.nunique) > 5

    binary_columns = X[categorical_mask.index[categorical_mask]].columns
    nominal_columns = X[categorical_mask.index[~categorical_mask]].columns
    numerical_columns = X[numerical_mask.index[numerical_mask]].columns

    return binary_columns, nominal_columns, numerical_columns

binary_columns, nominal_columns, numerical_columns = get_columns_by_category()

# Now we can create a column transformer pipeline

transformers = [('binary', OrdinalEncoder(), binary_columns),
                ('nominal', OneHotEncoder(), nominal_columns),
                ('numerical', StandardScaler(), numerical_columns)]

transformer_pipeline = ColumnTransformer(transformers, remainder='passthrough')

# Starified k cross validation
Kfold = StratifiedKFold(n_splits=5)

model = RandomForestClassifier(max_depth=7, 
                       min_samples_split=5, 
                       min_samples_leaf=5, random_state=42)

pipe = Pipeline([('transformer', transformer_pipeline), ('Random Forest Classifier', model)])

# Cross Validation

def cv_fit_models():
    train_acc_results = []
    cv_scores = {'Random Forest Classifier': []}
    cv_score = cross_validate(pipe,
                              X_train,
                              y_train,
                              scoring=scoring,
                              cv=Kfold,
                              return_train_score=True,
                              return_estimator=True)

    train_accuracy = cv_score['train_acc'].mean() * 100

    train_acc_results.append(train_accuracy)
    cv_scores['Random Forest Classifier'].append(cv_score)

    return np.array(train_acc_results), cv_scores

scoring = {'acc': 'accuracy'}

results, folds_scores = cv_fit_models()

# Pick the best fold for each model according to the highest test accuracy:

def pick_best_estimator():
    best_estimators = {'Random Forest Classifier': []}
    for key, model in folds_scores.items():
        best_acc_idx = np.argmax(model[0]['test_acc'])
        best_model = model[0]['estimator'][best_acc_idx]
        best_estimators[key].append(best_model)
    return best_estimators

best_estimators = pick_best_estimator()

modl =  best_estimators['Random Forest Classifier'][0]
#name = None
#pred = None




@app.route('/')
def main():
    return render_template('index.html')

@app.route('/form')
def main1():
    return render_template('form.html')

@app.route('/predict', methods= ['POST'])
def index():
    global name
    global pred
    name = request.form['name'].capitalize()
    age= request.form['age']
    stud_hr= request.form['stud_hr']
    employed= request.form['employed']
    h_disab= request.form['h_disab']
    ment_cond= request.form['ment_cond']
    social_hr= request.form['social_hr']
    fit_hr= request.form['fit_hr']
    wind= request.form['wind']
    dry_mouth= request.form['dry_mouth']
    positive= request.form['positive']
    breath_diff= request.form['breath_diff']
    initiate= request.form['initiate']
    tremb= request.form['tremb']
    worry= request.form['worry']
    look_fwd= request.form['look_fwd']
    down= request.form['down']
    enthus= request.form['enthus']
    life_mean= request.form['life_mean']
    scared= request.form['scared']
    arr = pd.DataFrame((np.array([[age,stud_hr,employed,h_disab,ment_cond,social_hr,fit_hr,wind,dry_mouth,
                positive,breath_diff,initiate,tremb,worry,look_fwd,down,enthus,
                life_mean,scared]])
        ), columns=X_train.columns)    
    pred= modl.predict(arr)

    return render_template('after.html', data=pred ,
       name = name)
    

@app.route('/music', methods= ['POST'])
def music():
    #global name
    #global pred
    music= request.form['music']
    #name = name 
    #data=pred
    return render_template('music.html', music=music, name = name,  data=pred)


if __name__ == '__main__':
    app.run(debug= False, use_reloader=False)
