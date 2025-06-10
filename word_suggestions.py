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
        for synset in wordnet.synsets(target_word):
            # Only use synsets matching the POS if we found one
            if word_pos and synset.pos() != word_pos:
                continue
                
            # Add both lemma names and similar words
            for lemma in synset.lemmas():
                if lemma.name().lower() != target_word.lower():
                    synonyms.add(lemma.name())
            
            # Add similar words for adjectives
            if synset.pos() == wordnet.ADJ:
                for similar in synset.similar_tos():
                    for lemma in similar.lemmas():
                        if lemma.name().lower() != target_word.lower():
                            synonyms.add(lemma.name())
        
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