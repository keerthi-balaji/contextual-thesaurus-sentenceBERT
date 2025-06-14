# Contextual Word Suggester

A smart thesaurus that suggests contextually appropriate word replacements using BERT embeddings and WordNet.

## Features

- Contextual word suggestions based on sentence meaning
- Maintains proper grammar and word forms
- Handles verbs, adjectives, adverbs, and nouns
- Provides confidence scores and word definitions
- Web interface for easy use

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/contextual-thesaurus-sentenceBERT.git
cd contextual-thesaurus-sentenceBERT
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the Flask application:
```bash
python app.py
```

Then open your browser and navigate to:
```
http://localhost:5000
```

Enter a sentence and select a word to get contextual suggestions.

## Technologies Used

- Python 3.x
- Sentence Transformers (BERT)
- NLTK (WordNet)
- Flask
- PyTorch

## Project Structure

```
contextual-thesaurus-sentenceBERT/
├── app.py              # Flask web application
├── word_suggestions.py # Core suggestion logic
├── static/            
│   ├── css/           # Styling
│   └── js/            # Frontend functionality
├── templates/         # HTML templates
└── requirements.txt   # Project dependencies
```