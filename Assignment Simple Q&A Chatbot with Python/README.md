# Custom AI Chatbot - MSCS-633 Assignment

## Installation & Setup

### Prerequisites
- Python 3.9+
- Django framework
- ChatterBot library
- spaCy for NLP processing

### Quick Start
```bash
# Clone repository
git clone https://github.com/harikiran2709/MSCS-633-Advance-Artificial-Intelligence

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run the chatbot
python chat_terminal.py
```

## Usage Examples

### Basic Conversation
```
You: Hello
Bot: Hi! I'm an AI chatbot. How can I assist you today?

You: What is artificial intelligence?
Bot: Artificial Intelligence (AI) is the simulation of human intelligence in machines.

You: Tell me about machine learning
Bot: Machine learning is a subset of AI that enables computers to learn without explicit programming.
```

## Technical Specifications

- **Framework**: Django + ChatterBot
- **Database**: SQLite with custom schema
- **NLP Engine**: spaCy for language processing
- **Architecture**: Object-oriented design
- **Features**: Custom training, logging, analytics

## Troubleshooting

### Common Issues
- **Import Errors**: Ensure all dependencies are installed
- **Training Issues**: Delete `db.sqlite3` to reset training data
- **spaCy Errors**: Download English model with provided command
