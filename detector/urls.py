"""proj URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import views
from django.contrib.auth.views import LoginView

urlpatterns = [
    path('', views.homepage),
    path('login/', LoginView.as_view(template_name='login.html'), name="login"),
    path('register/', views.register),
    path('login/userhome/', views.userHome),
    path('login/userhome/mode1/', views.mode1),
    path('login/userhome/mode2/', views.mode2),
    path('login/userhome/mode3/', views.mode3),
    path('login/userhome/takepictures/', views.takePictures),
    path('login/userhome/addlicense/', views.addLicense),
    path('login/userhome/takepictures/newface/', views.addFace),
    path('login/userhome/takepictures/done/', views.done),
]
