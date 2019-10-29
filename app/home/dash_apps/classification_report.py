import pickle

import dash
import dash_core_components as dcc
import dash_html_components as html
from visuals.custom_viz import CustomVisuals
from django_plotly_dash import DjangoDash

# Loading model attributes
model = pickle.load(open('../../models/model.pkl'))
feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Cabin_Class']
classes = ['not_survived', 'survived']

# Loading train_test data
X_train = np.load('../models/X_train.npy')
X_test = np.load('../models/X_test.npy')
y_train = np.load('../models/y_train.npy')
y_test = np.load('../models/y_test.npy')

cv = CustomVisuals(model=model, feature_names=feature_names, classes=classes)
app = DjangoDash('classificationreport')   # replaces dash.Dash

app.layout = html.Div([
    # dcc.RadioItems(
    #     id='dropdown-color',
    #     options=[{'label': c, 'value': c.lower()} for c in ['Red', 'Green', 'Blue']],
    #     value='red'
    # ),
    # html.Div(id='output-color'),
    # dcc.RadioItems(
    #     id='dropdown-size',
    #     options=[{'label': i,
    #               'value': j} for i, j in [('L','large'), ('M','medium'), ('S','small')]],
    #     value='medium'
    # ),
    # html.Div(id='output-size')

    dcc.Graph(
        id='classification-report',
        cv.generate_classification_report(X_test=X_test, y_test=y_test)
    )

])

# @app.callback(
#     dash.dependencies.Output('output-color', 'children'),
#     [dash.dependencies.Input('dropdown-color', 'value')])
# def callback_color(dropdown_value):
#     return "The selected color is %s." % dropdown_value

# @app.callback(
#     dash.dependencies.Output('output-size', 'children'),
#     [dash.dependencies.Input('dropdown-color', 'value'),
#      dash.dependencies.Input('dropdown-size', 'value')])
# def callback_size(dropdown_color, dropdown_size):
#     return "The chosen T-shirt is a %s %s one." %(dropdown_size,
#                                                   dropdown_color)