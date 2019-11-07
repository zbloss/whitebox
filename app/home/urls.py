from django.contrib import admin
from django.urls import path, include
from django.views.generic import TemplateView
from . import views

import home.dash_apps

urlpatterns = [
    path('', views.home, name='homepage'),
    path('analysis/', views.analysis, name='analysis'),
    path('explain/', views.explain, name='explain'),
    path('data/', views.data, name='data'),
    path('django_plotly_dash/', include('django_plotly_dash.urls')),
]
