from django.urls import path
from . import views

urlpatterns = [
    path("", views.predict_condition, name="predict_condition"),
    path("gemini-assistant/", views.gemini_assistant, name="gemini_assistant"),
]
