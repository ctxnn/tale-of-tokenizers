
from collections import Counter, defaultdict
import re
import json
import os

class BPETokenizer:
    """
    A subword tokenizer using Byte Pair Encoding (BPE) algorithm.
    """
    
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.merges = []                
        self.vocab = {}                 
        self.token_pattern = None       
        self.special_tokens = {         
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3
        }
    
    def get_stats(self, vocab):

        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs
    
    def merge_vocab(self, vocab, pair):

        new_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word, freq in vocab.items():
            # Replace space-separated pair with merged version
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = freq
            
        return new_vocab
    
    def fit(self, texts, num_symbols=None):

        if num_symbols:
            self.vocab_size = num_symbols
        
        words = []
        for text in texts:
            # Simple word tokenization by whitespace and keeping punctuation
            words.extend(re.findall(r'\w+|[^\w\s]', text))
        
        word_counts = Counter(words)
        
        vocab = {' '.join(word): count for word, count in word_counts.items()}
        
        chars = set()
        for word in word_counts.keys():
            for char in word:
                chars.add(char)
        
        num_merges = self.vocab_size - len(chars) - len(self.special_tokens)
        
        self.merges = []
        
        for i in range(num_merges):
            pair_counts = self.get_stats(vocab)
            
            if not pair_counts:
                break
                
            best_pair = max(pair_counts, key=pair_counts.get)
            
            self.merges.append(best_pair)
            
            vocab = self.merge_vocab(vocab, best_pair)
            
            if (i + 1) % 100 == 0 or i + 1 == num_merges:
                print(f"Merge {i + 1}/{num_merges}: {best_pair} -> {''.join(best_pair)}")
        
        self.vocab = {token: idx for token, idx in self.special_tokens.items()}
        next_id = len(self.special_tokens)
        
        for char in sorted(chars):
            if char not in self.vocab:
                self.vocab[char] = next_id
                next_id += 1
        
        for pair in self.merges:
            merged = ''.join(pair)
            if merged not in self.vocab:
                self.vocab[merged] = next_id
                next_id += 1
        
        tokens_by_length = sorted(
            [token for token in self.vocab if token not in self.special_tokens],
            key=len, 
            reverse=True
        )
        
        escaped_tokens = [re.escape(token) for token in tokens_by_length]
        
        self.token_pattern = re.compile('|'.join(escaped_tokens))
        
        print(f"Final vocabulary size: {len(self.vocab)} tokens")
        return self
    
    def tokenize(self, text):

        if not self.token_pattern:
            raise ValueError("Tokenizer hasn't been trained yet. Call fit() first.")
        
        tokens = self.token_pattern.findall(text)
        
        result = []
        position = 0
        for token in tokens:
            start = text.find(token, position)
            if start > position:
                unknown_chars = text[position:start]
                result.extend(['<UNK>'] * len(unknown_chars))
            
            result.append(token)
            position = start + len(token)
        
        if position < len(text):
            unknown_chars = text[position:]
            result.extend(['<UNK>'] * len(unknown_chars))
        
        return result
    
    def encode(self, text, add_special_tokens=False):

        tokens = self.tokenize(text)
        
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.special_tokens['<BOS>'])
        
        for token in tokens:
            token_id = self.vocab.get(token, self.special_tokens['<UNK>'])
            token_ids.append(token_id)
        
        if add_special_tokens:
            token_ids.append(self.special_tokens['<EOS>'])
            
        return token_ids
    
    def decode(self, token_ids, skip_special_tokens=True):

        id_to_token = {id: token for token, id in self.vocab.items()}
        
        tokens = []
        for id in token_ids:
            if skip_special_tokens and id in self.special_tokens.values():
                continue
                
            token = id_to_token.get(id, '<UNK>')
            tokens.append(token)
        
        return ''.join(tokens)
    
    def save(self, directory):

        os.makedirs(directory, exist_ok=True)
        
        with open(os.path.join(directory, 'vocab.json'), 'w') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        
        with open(os.path.join(directory, 'merges.txt'), 'w') as f:
            for pair in self.merges:
                f.write(f"{pair[0]} {pair[1]}\n")
        
        print(f"Tokenizer saved to {directory}")
    
    @classmethod
    def load(cls, directory):

        tokenizer = cls()
        
        with open(os.path.join(directory, 'vocab.json'), 'r') as f:
            tokenizer.vocab = json.load(f)
        
        for i in range(4):
            if str(i) in tokenizer.vocab:
                tokenizer.vocab[i] = tokenizer.vocab[str(i)]
                del tokenizer.vocab[str(i)]
        
        tokenizer.merges = []
        with open(os.path.join(directory, 'merges.txt'), 'r') as f:
            for line in f:
                pair = line.strip().split()
                if len(pair) == 2:
                    tokenizer.merges.append(tuple(pair))
        
        tokens_by_length = sorted(
            [token for token in tokenizer.vocab if token not in tokenizer.special_tokens.values()],
            key=len, 
            reverse=True
        )
        escaped_tokens = [re.escape(token) for token in tokens_by_length]
        tokenizer.token_pattern = re.compile('|'.join(escaped_tokens))
        
        print(f"Loaded tokenizer with {len(tokenizer.vocab)} tokens and {len(tokenizer.merges)} merges")
        return tokenizer

