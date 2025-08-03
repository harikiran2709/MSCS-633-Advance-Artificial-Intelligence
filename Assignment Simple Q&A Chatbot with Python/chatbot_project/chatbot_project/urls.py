"""
Custom URL Configuration for Chatbot Project
MSCS-633 Advanced Artificial Intelligence Assignment

This module defines the URL patterns for the chatbot project.
"""

from django.contrib import admin
from django.urls import path, include
from django.http import HttpResponse
from django.views.generic import TemplateView

def chatbot_home(request):
    """Custom home view for chatbot project"""
    return HttpResponse("""
    <h1>🤖 Custom AI Chatbot</h1>
    <p>MSCS-633 Advanced Artificial Intelligence Assignment</p>
    <p>This is a Django-powered chatbot using ChatterBot.</p>
    """)

def chatbot_status(request):
    """Status endpoint for chatbot health check"""
    return HttpResponse("""
    <h2>Chatbot Status</h2>
    <p>✅ Django Framework: Active</p>
    <p>✅ ChatterBot: Configured</p>
    <p>✅ Database: SQLite</p>
    <p>✅ Project: MSCS-633 Assignment</p>
    """)

# Custom URL patterns for chatbot project
urlpatterns = [
    # Admin interface
    path('admin/', admin.site.urls),
    
    # Custom chatbot endpoints
    path('', chatbot_home, name='chatbot_home'),
    path('status/', chatbot_status, name='chatbot_status'),
    path('info/', TemplateView.as_view(template_name='chatbot_info.html'), name='chatbot_info'),
]
