import os
import requests
import numpy as np
import pandas as pd

from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go

from yellowbrick.features import Rank2D, Rank1D
from yellowbrick.classifier import ROCAUC

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix


class CustomVisuals(object):
    
    def __init__(self, model, feature_names, classes):
        self.model = model
        self.feature_names = feature_names
        self.classes = classes
        

    def generate_classification_report(self, X_test, y_test, 
                                       zmin=0, zmax=1, annot=False, **kwargs):
        """
        Given the test data sets, produces a classification report and returns a plotly
        Heatmap figure.
        
        :param X_test: The withheld input features from the training data set.
        :param y_test: The withheld output target from the training data set.
        :param zmin: (optional) sets scale minimum. Defaults to 0.
        :param zmax: (optional) sets scale maxmimum. Defaults to 1.
        """
        cr = classification_report(y_true = y_test, y_pred = self.model.predict(X_test), output_dict=True)
        cr = pd.DataFrame(cr)
        
        fig_x = self.classes + ['accuracy', 'macro avg', 'weighted avg']
        fig_y = cr.index
        z_text = cr.values
        
        fig = go.Figure(
            go.Heatmap(
                z = cr.values,
                x = fig_x,
                y = fig_y,
                zmin = zmin if zmin is not None else zmin,
                zmax = zmax if zmax is not None else zmax,
                text = None if annot == False else z_text,
                **kwargs
            )
        )
        return fig


    def generate_confusion_matrix(self, X_test, y_test, **kwargs):
        """
        Given the test data sets, produces a confusion matrixs and returns a 
        plotly Heatmap figure.
        
        :param X_test: The withheld input features from the training data set.
        :param y_test: The withheld output target from the training data set.
        """
        
        cm = confusion_matrix(y_test, self.model.predict(X_test))
        
        fig_x = self.classes
        fig_y = self.classes
        fig_z = cm

        hovertext = list()
        for yi, yy in enumerate(fig_y):
            hovertext.append(list())
            for xi, xx in enumerate(fig_x):
                hovertext[-1].append(f'Predicted: {xx}<br />Actual: {yy}<br />Count: {fig_z[yi][xi]}')
      
        fig = go.Figure(
            go.Heatmap(
                z = fig_z, 
                x = fig_x,
                y = fig_y,
                hoverinfo='text',
                text=hovertext,
                **kwargs
            ),
            go.Layout(
                xaxis=go.layout.XAxis(
                    title=go.layout.xaxis.Title(text='Predicted', font=dict(size=24))
                ),
                yaxis=go.layout.YAxis(
                    title=go.layout.yaxis.Title(text='Actuals', font=dict(size=24))
                )
            
            )
        )
        return fig
    
    
    def generate_rank_2d(self, X, algorithm='pearson', **kwargs):
        """
        Given the entire (train+test) input features, returns a plotly
        Heatmap figure showing the feature x feature correlation.
        
        :param X: the input features to the model
        :param algorithm: the algorithm to calculate the importance with 
                          (pearson, covariance, spearman, kendalltau)
        """
        
        visualizer = Rank2D(algorithm=algorithm)
        visualizer.fit_transform(X)

        # values
        ranks_ = visualizer.ranks_

        #grabbing feature shape
        feats = ranks_.shape[0]
        
        # zero-ing out one of the diagonals features
        iu = np.triu_indices(feats, )
        ranks_[iu] = 0
        
        fig = go.Figure(
            go.Heatmap(
                z = ranks_,
                x = self.feature_names,
                y = np.flip(self.feature_names),
                **kwargs
            )
        )
        
        return fig
    
    
    def generate_rank_1d(self, X, y, algorithm='shapiro', **kwargs):
        """
        Given the entire (train+test) input and target features, returns 
        a plotly figure showing the feature correlation.
        
        :param X: the input features to the model
        :param y: the target feature
        """
        
        visualizer = Rank1D(algorithm='shapiro')

        visualizer.fit(X, y)           # Fit the data to the visualizer
        visualizer.transform(X)        # Transform the data
        
        fig = go.Figure(
            go.Bar(
                x = visualizer.ranks_,
                y = self.feature_names,  #visualizer.features_,
                orientation='h'
            )
        )
        return fig
        

    def generate_rad_viz(self, X, y, **kwargs):
        """
        Given the entire feature-space, X, and target-space, y, 
        generates a RadViz plotly object.
        
        :param X: the input features to the model.
        :param y: the target variables to the model.
        """
        
        # grabbing rows, columns, to later build out the radviz data set
        nrows, ncols = X.shape
        to_plot = {label: [[], []] for label in self.classes}
        
        # locations of where to plot the feature names.
        class_points = np.array(
            [
                (np.cos(t), np.sin(t)) for t in [2.0 * np.pi * (i / float(ncols)) for i in range(ncols)]
            ]
        )
        
        # converting from Data Frame to numpy array if need be
        sc = MinMaxScaler(feature_range=(0,1))    
        X_scaled = sc.fit_transform(X)
        
        # Compute the locations of the scatter plot for each class
        for i, row in enumerate(X_scaled):
            row_ = np.repeat(np.expand_dims(row, axis=1), 2, axis=1)
            xy = (class_points * row_).sum(axis=0) / row.sum()
            label = self.classes[y[i]]
            #label = 'not_survived' if y.values[i] == 0 else 'survived'  #self._label_encoder[y[i]]
            to_plot[label][0].append(xy[0])
            to_plot[label][1].append(xy[1])

        layout = go.Layout(yaxis = dict(
                            scaleanchor = "x",
                            #scaleratio = 1,
                            range=[-1.1, 1.1]
                        ),
                        xaxis=dict(
                            range=[-1.1, 1.1],
                            #scaleratio = 1
                        ),
                )

        fig = go.Figure(layout=layout)
            
        for lab in self.classes:
            trace = go.Scatter(
                x = to_plot[lab][0],
                y = to_plot[lab][1],
                mode = 'markers', 
                name = f'{lab}'
            )
            fig.add_trace(trace)
        

        class_trace = go.Scatter(
            x = [class_points[i][0] for i in range(len(class_points))] + [class_points[0][0]],
            y = [class_points[i][1] for i in range(len(class_points))] + [class_points[0][1]],
            name = 'Features',
            mode = 'lines+text',
            text = [self.feature_names[i] for i in range(len(self.feature_names))]
        )
        
        fig.add_trace(class_trace)

        return fig
    

    def generate_roc_auc(self, X_train, y_train, X_test, y_test, **kwargs):
        """
        Given the training and testing sets, computes the ROC AUC metrics
        for the given model and returns a ROC AUC plotly figure.
        
        :param X_train: the training feature set.
        :param y_train: the training target set.
        :param X_test: the testing feature set.
        :param y_test: the testing target set.
        """
        
        visualizer = ROCAUC(self.model, classes=self.classes)

        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        
        roc_data = visualizer.tpr
        
        layout = go.Layout(yaxis = dict(
                                scaleratio = 1,
                                range=[-.1, 1]
                            ),
                            xaxis=dict(
                                range=[-.1, 1],
                                #scaleratio = 1
                            ),
                        )

        fig = go.Figure(layout=layout)

        for tr in roc_data.keys():
            trace = go.Scatter(
                x = [i / len(roc_data[tr]) for i in range(len(roc_data[tr]))],
                y = roc_data[tr],
                name = f'{tr}' if type(tr) != int else f'{self.classes[tr]}',
                line = dict(shape = 'hv')
            )
            fig.add_trace(trace)

        lin_line = go.Scatter(
            x = [0,1],
            y = [0,1],
            name = 'linear_line',
            line = dict(dash='dash')
        )

        fig.add_trace(lin_line)
        return fig
    

    def generate_pca_2d(self, X, y, scale=True, **kwargs):
        """
        Given the entire feature and target sets, performs a 2D Principal
        Component Analysis and returns a plotly figure
        
        :param X: model feature set.
        :param y: model target set.
        :param scale: bool indicating if you want to perform Standard Scaling.
        """
        
        if scale:
            X = StandardScaler().fit_transform(X)

        model = PCA(n_components = 2)
        model.fit(X)
        features = model.transform(X) 
        
        layout = go.Layout(
            xaxis=dict(title='PC<sub>1</sub>'),
            yaxis=dict(title='PC<sub>2</sub>')
        )
        
        fig = go.Figure(layout=layout)
        
        for class_ in range(len(self.classes)):
            temp_x, temp_y = features[y == class_].T
            
            trace = go.Scatter(
                x = temp_x,
                y = temp_y,
                name = f'{self.classes[class_]}',
                mode = 'markers'
            )
            
            fig.add_trace(trace)
    
        return fig
    
    
    def generate_pca_3d(self, X, y, scale=True, **kwargs):
        """
        Given the entire feature and target sets, performs a 3D Principal
        Component Analysis and returns a plotly figure
        
        :param X: model feature set.
        :param y: model target set.
        :param scale: bool indicating if you want to perform Standard Scaling.
        """
        
        if scale:
            X = StandardScaler().fit_transform(X)

        model = PCA(n_components = 3)
        model.fit(X)
        features = model.transform(X) 
        
        layout = go.Layout(height=600)

        fig = go.Figure(layout=layout)
        
        for class_ in range(len(self.classes)):
            temp_x, temp_y, temp_z = features[y == class_].T
            
            trace = go.Scatter3d(
                x = temp_x,
                y = temp_y,
                z = temp_z,
                name = f'{self.classes[class_]}',
                mode = 'markers'
            )
            
            fig.add_trace(trace)
            
        
        return fig