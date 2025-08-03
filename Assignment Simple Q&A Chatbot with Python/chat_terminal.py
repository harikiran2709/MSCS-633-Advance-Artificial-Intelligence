"""
Custom Q&A Chatbot Implementation
MSCS-633 Advanced Artificial Intelligence Assignment

This implementation creates a unique chatbot with custom features
including conversation history, custom training data, and enhanced response handling.
"""

import os
import django
import json
import random
from datetime import datetime

# Configure Django for ChatterBot
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chatbot_project.chatbot_project.settings')
django.setup()

from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer, ListTrainer

class CustomChatbot:
    """Custom chatbot implementation with unique features"""
    
    def __init__(self):
        self.conversation_history = []
        self.custom_responses = {
            'greeting': [
                "Hello! I'm your AI assistant. How can I help you today?",
                "Hi there! Nice to meet you. What's on your mind?",
                "Greetings! I'm here to chat with you. How are you doing?"
            ],
            'farewell': [
                "Goodbye! It was nice chatting with you!",
                "See you later! Have a great day!",
                "Take care! Come back anytime!"
            ]
        }
        
        # Initialize ChatterBot with custom configuration
        self.bot = ChatBot(
            name='CustomAIBot',
            storage_adapter='chatterbot.storage.SQLStorageAdapter',
            database_uri='sqlite:///db.sqlite3',
            logic_adapters=[
                'chatterbot.logic.BestMatch',
                'chatterbot.logic.MathematicalEvaluation'
            ],
            read_only=False
        )
    
    def train_with_custom_data(self):
        """Train the bot with custom conversation data"""
        print("Training with custom conversation data...")
        
        # Custom training conversations
        custom_conversations = [
            [
                "What is artificial intelligence?",
                "Artificial Intelligence (AI) is the simulation of human intelligence in machines.",
                "How does AI work?",
                "AI works by processing data, learning patterns, and making decisions based on algorithms."
            ],
            [
                "Tell me about machine learning",
                "Machine learning is a subset of AI that enables computers to learn without explicit programming.",
                "What are the types of machine learning?",
                "The main types are supervised learning, unsupervised learning, and reinforcement learning."
            ],
            [
                "Hello",
                "Hi! I'm an AI chatbot. How can I assist you today?",
                "How are you?",
                "I'm functioning perfectly! How about you?"
            ]
        ]
        
        # Train with custom data
        list_trainer = ListTrainer(self.bot)
        for conversation in custom_conversations:
            list_trainer.train(conversation)
        
        # Also train with standard corpus
        corpus_trainer = ChatterBotCorpusTrainer(self.bot)
        corpus_trainer.train("chatterbot.corpus.english.greetings")
        corpus_trainer.train("chatterbot.corpus.english.conversations")
        
        print("Custom training completed!")
    
    def get_custom_response(self, user_input):
        """Get response with custom logic"""
        user_input_lower = user_input.lower()
        
        # Check for custom patterns
        if any(word in user_input_lower for word in ['hello', 'hi', 'hey']):
            return random.choice(self.custom_responses['greeting'])
        elif any(word in user_input_lower for word in ['bye', 'goodbye', 'exit', 'quit']):
            return random.choice(self.custom_responses['farewell'])
        
        # Use ChatterBot for other responses
        try:
            response = self.bot.get_response(user_input)
            return str(response)
        except Exception as e:
            return "I'm not sure how to respond to that. Could you rephrase?"
    
    def log_conversation(self, user_input, bot_response):
        """Log conversation for analysis"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.conversation_history.append({
            'timestamp': timestamp,
            'user_input': user_input,
            'bot_response': bot_response
        })
    
    def display_stats(self):
        """Display conversation statistics"""
        if self.conversation_history:
            print(f"\nConversation Statistics:")
            print(f"Total exchanges: {len(self.conversation_history)}")
            print(f"Session duration: {self.conversation_history[-1]['timestamp']}")

def run_chatbot():
    """Main function to run the custom chatbot"""
    print("=" * 60)
    print("🤖 Custom AI Chatbot - MSCS-633 Assignment")
    print("=" * 60)
    
    # Initialize chatbot
    chatbot = CustomChatbot()
    
    # Train the bot
    chatbot.train_with_custom_data()
    
    print("\nChatbot is ready! Start chatting (type 'exit' to quit)")
    print("-" * 60)
    
    # Main conversation loop
    while True:
        try:
            # Get user input
            user_message = input("You: ").strip()
            
            # Check for exit command
            if user_message.lower() in ['exit', 'quit', 'bye']:
                final_response = chatbot.get_custom_response(user_message)
                print(f"Bot: {final_response}")
                chatbot.display_stats()
                break
            
            # Skip empty input
            if not user_message:
                continue
            
            # Get bot response
            bot_response = chatbot.get_custom_response(user_message)
            print(f"Bot: {bot_response}")
            
            # Log the conversation
            chatbot.log_conversation(user_message, bot_response)
            
        except KeyboardInterrupt:
            print("\nBot: Goodbye! Thanks for chatting!")
            chatbot.display_stats()
            break
        except Exception as e:
            print(f"Bot: Sorry, I encountered an error: {e}")

if __name__ == "__main__":
    run_chatbot()