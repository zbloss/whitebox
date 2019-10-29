from django.shortcuts import render
from django.http import HttpResponse
#from .dash_apps import classification_report

from .dash_apps import *


def index(request):
    return HttpResponse("Hello, world. You're at the index.")

def home(request):
    return render(request, 'index.html')

def test(request):
    return render(request, 'test.html') # , context=dash_context)