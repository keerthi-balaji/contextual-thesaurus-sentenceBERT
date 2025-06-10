from flask import Flask, render_template, request, jsonify
from word_suggestions import ContextualWordSuggester

app = Flask(__name__)
suggester = ContextualWordSuggester()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_suggestions', methods=['POST'])
def get_suggestions():
    data = request.get_json()
    sentence = data.get('sentence', '')
    target_word = data.get('target_word', '')
    
    if not sentence or not target_word:
        return jsonify({'error': 'Missing sentence or target word'})
        
    if target_word not in sentence:
        return jsonify({'error': 'Target word not found in sentence'})
        
    suggestions = suggester.get_suggestions(sentence, target_word)
    
    if not suggestions:
        return jsonify({'error': 'No suggestions found'})
        
    return jsonify({
        'suggestions': [
            {
                'word': word,
                'confidence': float(score),
                'meaning': meaning
            }
            for word, score, meaning in suggestions
        ]
    })

if __name__ == '__main__':
    app.run(debug=True)