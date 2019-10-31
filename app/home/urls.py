from django.contrib import admin
from django.urls import path, include
from django.views.generic import TemplateView
from . import views

import home.dash_apps

urlpatterns = [
    path('', views.home, name='homepage'),
    path('analysis/', views.analysis, name='analysis'),
    path('new_test/', TemplateView.as_view(template_name='test.html'), name="demo-one"),
    path('plain/', TemplateView.as_view(template_name='plain.html'), name='plain_test'),

    path('django_plotly_dash/', include('django_plotly_dash.urls')),
]
