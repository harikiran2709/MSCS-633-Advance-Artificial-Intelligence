"""
Custom ASGI Configuration for Chatbot Project
MSCS-633 Advanced Artificial Intelligence Assignment

This module contains the ASGI callable as a module-level variable named `application`.
"""

import os
import sys
from pathlib import Path

# Add the project directory to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Set the Django settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chatbot_project.settings')

# Import Django ASGI application
from django.core.asgi import get_asgi_application

# Create the ASGI application object
application = get_asgi_application()

# Custom ASGI middleware for chatbot project
def chatbot_asgi_middleware(scope, receive, send):
    """Custom ASGI middleware for chatbot-specific functionality"""
    # Add custom headers for chatbot project
    if scope['type'] == 'http':
        scope['headers'].append((b'x-chatbot-project', b'MSCS-633-Assignment'))
        scope['headers'].append((b'x-chatbot-version', b'1.0.0'))
    
    return application(scope, receive, send)

# Use custom middleware if needed
# application = chatbot_asgi_middleware
