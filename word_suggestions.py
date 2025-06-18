from sentence_transformers import SentenceTransformer
from nltk.corpus import wordnet
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import nltk
from typing import List, Tuple
import torch

class ContextualWordSuggester:
    def __init__(self):
        # Use a simpler model that works well with emotions and common adjectives
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
        # Download required NLTK data
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
        
        # POS mapping
        self.pos_map = {
            'JJ': wordnet.ADJ,
            'JJR': wordnet.ADJ,
            'JJS': wordnet.ADJ,
            'RB': wordnet.ADV,
            'RBR': wordnet.ADV,
            'RBS': wordnet.ADV,
            'VB': wordnet.VERB,
            'VBD': wordnet.VERB,
            'VBG': wordnet.VERB,
            'VBN': wordnet.VERB,
            'VBP': wordnet.VERB,
            'VBZ': wordnet.VERB,
            'NN': wordnet.NOUN,
            'NNS': wordnet.NOUN,
        }
        
        # Add verb forms dictionary
        self.verb_forms = {
            'VBD': 'past',       
            'VBN': 'past',       
            'VBG': 'gerund',     
            'VBZ': 'present',    
            'VBP': 'present',    
            'VB': 'base'         
        }

    def _adjust_verb_form(self, word: str, original_pos: str) -> str:
        """Adjust verb to match the original word's tense"""
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        
        base_form = lemmatizer.lemmatize(word, 'v')
        
        if original_pos in ['VBD', 'VBN']:  # Past tense/participle
            if base_form.endswith('e'):
                return base_form + 'd'
            return base_form + 'ed'
        elif original_pos == 'VBG':  # Present participle
            if base_form.endswith('e'):
                return base_form[:-1] + 'ing'
            return base_form + 'ing'
        elif original_pos == 'VBZ':  # 3rd person singular
            if base_form.endswith(('s', 'sh', 'ch', 'x', 'z')):
                return base_form + 'es'
            elif base_form.endswith('y'):
                return base_form[:-1] + 'ies'
            return base_form + 's'
        
        return base_form

    def get_suggestions(self, sentence: str, target_word: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        # Get word's POS in context
        tokens = word_tokenize(sentence)
        pos_tags = pos_tag(tokens)
        word_pos = None
        
        # Find the POS tag for the target word
        for word, pos in pos_tags:
            if word.lower() == target_word.lower():
                word_pos = self.pos_map.get(pos)
                break
        
        # Get synonyms from WordNet
        synonyms = set()
        target_word_lower = target_word.lower()
        word_pos = None
        original_pos = None
        word_index = -1

        # Find target word position and POS
        for i, (word, pos) in enumerate(pos_tags):
            if word.lower() == target_word_lower:
                word_pos = self.pos_map.get(pos)
                original_pos = pos
                word_index = i
                break

        # 1. Try direct synonyms (allow up to 3-word phrases)
        for synset in wordnet.synsets(target_word):
            if word_pos and synset.pos() != word_pos:
                continue
            for lemma in synset.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if len(synonym.split()) <= 3 and synonym.lower() != target_word_lower:
                    synonyms.add(synonym)

        # 2. Fallback for adverbs ending in 'ly'
        if not synonyms and target_word_lower.endswith('ly'):
            base = target_word_lower[:-2]
            for synset in wordnet.synsets(base):
                if synset.pos() in [wordnet.ADJ, wordnet.ADV]:
                    for lemma in synset.lemmas():
                        synonym = lemma.name().replace('_', ' ')
                        if len(synonym.split()) <= 3 and synonym.lower() != base:
                            synonyms.add(synonym + 'ly')

        # 3. Supplement for tricky words like "boring"
        if not synonyms and target_word_lower == 'boring':
            extra_boring = ['tedious', 'dull', 'monotonous', 'tiresome', 'uninteresting', 'wearisome', 'mind-numbing']
            synonyms.update(extra_boring)

        if not synonyms:
            return []

        # Create candidate sentences
        candidate_sentences = []
        candidate_words = []
        
        for synonym in synonyms:
            # Replace underscores with spaces and skip multi-word expressions
            synonym = synonym.replace('_', ' ')
            if len(synonym.split()) > 1:
                continue
                
            new_sentence = sentence.replace(target_word, synonym)
            candidate_sentences.append(new_sentence)
            candidate_words.append(synonym)

        if not candidate_sentences:
            return []
            
        # Get embeddings and calculate similarities
        sentence_embedding = self.model.encode([sentence], convert_to_tensor=True)
        candidate_embeddings = self.model.encode(candidate_sentences, convert_to_tensor=True)
        
        # Calculate cosine similarities
        similarities = torch.nn.functional.cosine_similarity(
            candidate_embeddings,
            sentence_embedding.repeat(len(candidate_embeddings), 1)
        )
        
        # Create scored pairs
        scored_pairs = list(zip(candidate_words, similarities.tolist()))
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k suggestions with meanings
        suggestions = []
        seen_words = set()
        
        for word, score in scored_pairs:
            if word in seen_words:
                continue
                
            meaning = self._get_best_meaning(word, word_pos)
            suggestions.append((word, score, meaning))
            seen_words.add(word)
            
            if len(suggestions) >= top_k:
                break
                
        return suggestions

    def _get_best_meaning(self, word: str, pos=None) -> str:
        synsets = wordnet.synsets(word)
        if not synsets:
            return "No definition found"
            
        if pos:
            matching_synsets = [s for s in synsets if s.pos() == pos]
            if matching_synsets:
                return matching_synsets[0].definition()
                
        return synsets[0].definition()