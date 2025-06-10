from word_suggestions import ContextualWordSuggester

def main():
    suggester = ContextualWordSuggester()
    
    # Example usage
    sentence = "The small dog ran quickly across the field"
    target_word = "small"
    
    suggestions = suggester.get_suggestions(sentence, target_word)
    
    print(f"\nSuggestions for replacing '{target_word}' in:\n'{sentence}'\n")
    for word, score, meaning in suggestions:
        print(f"Word: {word}")
        print(f"Confidence: {score:.3f}")
        print(f"Meaning: {meaning}\n")

if __name__ == "__main__":
    main()