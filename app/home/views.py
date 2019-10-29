from django.shortcuts import render
from django.http import HttpResponse
import home.dash_apps as da

def index(request):
    return HttpResponse("Hello, world. You're at the index.")

def home(request):
    return render(request, 'index.html')

def test(request):
    return render(request, 'test.html')