from django.shortcuts import render
from django.http import HttpResponse
#from .dash_apps import classification_report

from .dash_apps import *


def index(request):
    return HttpResponse("Hello, world. You're at the index.")

def home(request):
    return render(request, 'home.html')

def analysis(request):
    return render(request, 'analysis.html')

def explain(request):
    return render(request, 'explain.html')

def data(request):
    return render(request, 'data.html')

def explanation(request):
    return render(request, 'explanation.html')