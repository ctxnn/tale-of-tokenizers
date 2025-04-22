import re
import string
from collections import Counter

class WordTokenizer:
    """
    A word-level tokenizer that splits text into words and punctuation.
    """
    
    def __init__(self, min_freq=1, max_vocab_size=None, lowercase=True):
        self.word_to_id = {}      
        self.id_to_word = {}       
        self.vocab_size = 0        
        self.min_freq = min_freq   
        self.max_vocab_size = max_vocab_size  
        self.lowercase = lowercase  
        

        self.special_tokens = {
            '<PAD>': 0,   
            '<UNK>': 1,   
            '<BOS>': 2,  
            '<EOS>': 3   
        }
        
        
        for token, idx in self.special_tokens.items():
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token
        
        self.vocab_size = len(self.special_tokens)
        
        self.pattern = r'\b\w+\b|[' + re.escape(string.punctuation) + r']'
    
    def tokenize(self, text):

        if self.lowercase:
            text = text.lower()
        
        tokens = re.findall(self.pattern, text)
        return tokens
    
    def fit(self, texts):
        self.id_to_word = {idx: token for token, idx in self.special_tokens.items()}
        self.vocab_size = len(self.special_tokens)
        word_counts = Counter()  # <-- Initialize Counter here
        
        for text in texts:
            tokens = self.tokenize(text)
            word_counts.update(tokens)
        
        word_counts = {word: count for word, count in word_counts.items() 
                      if count >= self.min_freq}
        
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        if self.max_vocab_size:
            sorted_words = sorted_words[:self.max_vocab_size - len(self.special_tokens)]
        
        for word, count in sorted_words:
            if word not in self.word_to_id:  # Skip words that are already special tokens
                self.word_to_id[word] = self.vocab_size
                self.id_to_word[self.vocab_size] = word
                self.vocab_size += 1
        
        print(f"Vocabulary size: {self.vocab_size} words")
        return self
    
    def encode(self, text, add_special_tokens=False):
        tokens = self.tokenize(text)
        
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.special_tokens['<BOS>'])
        
        for token in tokens:
            token_id = self.word_to_id.get(token, self.special_tokens['<UNK>'])
            token_ids.append(token_id)
        
        if add_special_tokens:
            token_ids.append(self.special_tokens['<EOS>'])
            
        return token_ids
    
    def decode(self, token_ids, skip_special_tokens=True):

        words = []
        for id in token_ids:
            if skip_special_tokens and id in [self.special_tokens['<PAD>'], 
                                             self.special_tokens['<BOS>'], 
                                             self.special_tokens['<EOS>']]:
                continue
                
            word = self.id_to_word.get(id, self.special_tokens['<UNK>'])
            words.append(word)
        
  
        return ' '.join(words)