async function getSuggestions() {
    const sentence = document.getElementById('sentence').value;
    const targetWord = document.getElementById('target-word').value;
    const resultsDiv = document.getElementById('results');
    
    if (!sentence || !targetWord) {
        resultsDiv.innerHTML = '<div class="error">Please enter both a sentence and a word to replace.</div>';
        return;
    }
    
    try {
        const response = await fetch('/get_suggestions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                sentence: sentence,
                target_word: targetWord
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            resultsDiv.innerHTML = `<div class="error">${data.error}</div>`;
            return;
        }
        
        let html = '<h2>Suggestions:</h2>';
        data.suggestions.forEach(suggestion => {
            html += `
                <div class="suggestion-card">
                    <div class="word">${suggestion.word}</div>
                    <div class="confidence">Confidence: ${(suggestion.confidence * 100).toFixed(1)}%</div>
                    <div class="meaning">Meaning: ${suggestion.meaning}</div>
                </div>
            `;
        });
        
        resultsDiv.innerHTML = html;
    } catch (error) {
        resultsDiv.innerHTML = '<div class="error">An error occurred while getting suggestions.</div>';
        console.error('Error:', error);
    }
}