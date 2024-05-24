from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_image, name='upload_image'),
    path('classify/', views.classify_image, name='classify_image'),
]
