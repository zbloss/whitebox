import pickle
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html

from .visuals.custom_viz import CustomVisuals
from django_plotly_dash import DjangoDash

# Loading model attributes
print('Loading model...')
model = pickle.load(open('../models/model.pkl', 'rb'))
feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Cabin_Class']
classes = ['not_survived', 'survived']

# Loading train_test data
print('Loading data...')
X_train = np.load('../models/X_train.npy')
X_test = np.load('../models/X_test.npy')
y_train = np.load('../models/y_train.npy')
y_test = np.load('../models/y_test.npy')

cv = CustomVisuals(model=model, feature_names=feature_names, classes=classes)

dash_name = 'classification_report'
app_name = 'whitebox'

app = DjangoDash(name=dash_name, app_name=app_name)
app.layout = html.Div([
    dcc.Graph(
        id = 'classification_report',
       figure =  cv.generate_classification_report(X_test, y_test)
    )
])

