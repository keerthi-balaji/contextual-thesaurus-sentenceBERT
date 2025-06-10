from sentence_transformers import SentenceTransformer
from nltk.corpus import wordnet
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import nltk
from typing import List, Tuple
import torch

class ContextualWordSuggester:
    def __init__(self):
        try:
            # Using a simpler model that's publicly available
            self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

        # Download required NLTK data
        try:
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('punkt', quiet=True)
        except Exception as e:
            print(f"Error downloading NLTK data: {str(e)}")
            raise

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
            'NNP': wordnet.NOUN,
            'NNPS': wordnet.NOUN,
        }

    def get_word_pos(self, sentence: str, target_word: str) -> str:
        """Get the part of speech of the target word in context"""
        tokens = word_tokenize(sentence)
        pos_tags = pos_tag(tokens)
        for word, pos in pos_tags:
            if word.lower() == target_word.lower():
                return self.pos_map.get(pos, None)
        return None

    def get_suggestions(self, sentence: str, target_word: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        # Get word's POS in context
        target_pos = self.get_word_pos(sentence, target_word)
        
        # Get quality synonyms from WordNet
        synonyms = set()
        for synset in wordnet.synsets(target_word):
            # Only consider words with matching POS
            if target_pos and synset.pos() != target_pos:
                continue
            
            for lemma in synset.lemmas():
                # Filter out multi-word expressions and proper nouns
                word = lemma.name()
                if '_' not in word and not word[0].isupper():
                    synonyms.add(word)
        
        if not synonyms:
            return []

        # Remove the target word from synonyms
        synonyms.discard(target_word)
        
        # Create candidate sentences
        candidate_sentences = []
        candidate_words = []
        
        for synonym in synonyms:
            new_sentence = sentence.replace(target_word, synonym)
            candidate_sentences.append(new_sentence)
            candidate_words.append(synonym)

        if not candidate_sentences:
            return []
            
        # Get embeddings and calculate similarities
        original_embedding = self.model.encode([sentence], convert_to_tensor=True)
        candidate_embeddings = self.model.encode(candidate_sentences, convert_to_tensor=True)
        
        # Calculate cosine similarities
        similarities = torch.nn.functional.cosine_similarity(
            candidate_embeddings,
            original_embedding.repeat(len(candidate_embeddings), 1)
        )
        
        # Create scored pairs
        scored_pairs = list(zip(candidate_words, similarities.tolist()))
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Get meanings and create final suggestions
        suggestions = []
        for word, score in scored_pairs[:top_k]:
            meaning = self._get_best_meaning(word, target_pos)
            suggestions.append((word, score, meaning))
            
        return suggestions

    def _get_best_meaning(self, word: str, pos=None) -> str:
        """Get the most relevant definition based on POS"""
        synsets = wordnet.synsets(word)
        if not synsets:
            return "No definition found"
            
        # Filter by POS if available
        if pos:
            matching_synsets = [s for s in synsets if s.pos() == pos]
            if matching_synsets:
                return matching_synsets[0].definition()
                
        return synsets[0].definition()