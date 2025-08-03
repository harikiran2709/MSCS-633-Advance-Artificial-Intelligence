"""
Custom WSGI Configuration for Chatbot Project
MSCS-633 Advanced Artificial Intelligence Assignment

This module contains the WSGI callable as a module-level variable named `application`.
"""

import os
import sys
from pathlib import Path

# Add the project directory to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Set the Django settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chatbot_project.settings')

# Import Django WSGI application
from django.core.wsgi import get_wsgi_application

# Create the WSGI application object
application = get_wsgi_application()

# Custom WSGI middleware for chatbot project
def chatbot_wsgi_middleware(environ, start_response):
    """Custom middleware for chatbot-specific functionality"""
    # Add custom headers for chatbot project
    environ['HTTP_X_CHATBOT_PROJECT'] = 'MSCS-633-Assignment'
    environ['HTTP_X_CHATBOT_VERSION'] = '1.0.0'
    
    return application(environ, start_response)

# Use custom middleware if needed
# application = chatbot_wsgi_middleware
