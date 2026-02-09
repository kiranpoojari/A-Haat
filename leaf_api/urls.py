from django.urls import path
from .views import LeafHealthAPIView, ArecaCoconutAPIView

urlpatterns = [
    path("leaf-health/", LeafHealthAPIView.as_view()),
    path("areca-coconut/", ArecaCoconutAPIView.as_view()),
]
