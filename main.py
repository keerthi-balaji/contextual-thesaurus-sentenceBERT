from word_suggestions import ContextualWordSuggester

def main():
    suggester = ContextualWordSuggester()
    
    print("Welcome to the Contextual Word Suggester!")
    print("Enter a sentence and the word you'd like to replace.\n")
    
    while True:
        # Get user input
        sentence = input("Enter your sentence (or 'q' to quit): ").strip()
        if sentence.lower() == 'q':
            break
            
        target_word = input("Enter the word to replace: ").strip()
        
        # Validate input
        if target_word not in sentence:
            print(f"Error: '{target_word}' not found in the sentence. Please try again.\n")
            continue
            
        # Get suggestions
        suggestions = suggester.get_suggestions(sentence, target_word)
        
        if not suggestions:
            print(f"\nNo suggestions found for '{target_word}' in this context.\n")
            continue
            
        # Display results
        print(f"\nSuggestions for replacing '{target_word}' in:\n'{sentence}'\n")
        for word, score, meaning in suggestions:
            print(f"Word: {word}")
            print(f"Confidence: {score:.3f}")
            print(f"Meaning: {meaning}\n")
        
        print("-" * 50 + "\n")

if __name__ == "__main__":
    main()