#!/usr/bin/env python
"""
Custom Django Management Script for Chatbot Project
MSCS-633 Advanced Artificial Intelligence Assignment

This script provides enhanced Django management capabilities
with custom chatbot-specific commands and error handling.
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    """Configure Django environment with custom settings"""
    # Set the Django settings module
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chatbot_project.settings')
    
    # Add the project directory to Python path
    project_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(project_root))

def check_django_installation():
    """Verify Django is properly installed"""
    try:
        import django
        print(f"✓ Django {django.get_version()} is installed")
        return True
    except ImportError:
        print("✗ Django is not installed. Please install it with: pip install django")
        return False

def check_chatbot_dependencies():
    """Verify ChatterBot dependencies are available"""
    try:
        from chatterbot import ChatBot
        print("✓ ChatterBot is available")
        return True
    except ImportError:
        print("✗ ChatterBot is not installed. Please install it with: pip install chatterbot")
        return False

def run_custom_commands():
    """Execute custom chatbot-specific commands"""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'chatbot':
            print("🤖 Starting Chatbot Management...")
            print("Available chatbot commands:")
            print("  python manage.py chatbot train    - Train the chatbot")
            print("  python manage.py chatbot test     - Test the chatbot")
            print("  python manage.py chatbot reset    - Reset chatbot database")
            return True
        
        elif command == 'setup':
            print("🔧 Setting up Chatbot Project...")
            setup_environment()
            
            if check_django_installation() and check_chatbot_dependencies():
                print("✓ All dependencies are properly configured")
                return True
            else:
                print("✗ Some dependencies are missing")
                return False
    
    return False

def execute_django_command():
    """Execute standard Django management commands"""
    try:
        from django.core.management import execute_from_command_line
        execute_from_command_line(sys.argv)
    except ImportError as exc:
        print("❌ Django Import Error:")
        print("   - Make sure Django is installed: pip install django")
        print("   - Check if virtual environment is activated")
        print("   - Verify PYTHONPATH is correctly set")
        raise ImportError(
            "Django is not properly installed or configured. "
            "Please check your installation and virtual environment."
        ) from exc

def main():
    """Enhanced main function with custom chatbot management"""
    # Setup environment
    setup_environment()
    
    # Check for custom commands first
    if run_custom_commands():
        return
    
    # Verify dependencies before running Django commands
    if not check_django_installation():
        sys.exit(1)
    
    # Execute Django management commands
    execute_django_command()

if __name__ == '__main__':
    main()
