class CharacterTokenizer:
    """
    A simple character-level tokenizer that splits text into individual characters.
    """
    
    def __init__(self):
        self.char_to_id = {}  
        self.id_to_char = {}  
        self.vocab_size = 0   
    
    def fit(self, texts):

        self.char_to_id = {}
        self.id_to_char = {}
        
        unique_chars = set()
        for text in texts:
            for char in text:
                unique_chars.add(char)
        
        for i, char in enumerate(sorted(unique_chars)):
            self.char_to_id[char] = i
            self.id_to_char[i] = char
        
        self.vocab_size = len(self.char_to_id)
        print(f"vocab size: {self.vocab_size} characters")
        
        return self
    
    def encode(self, text):
        return [self.char_to_id.get(char, self.char_to_id.get('<UNK>', -1)) for char in text]
    
    def decode(self, token_ids):
        return ''.join([self.id_to_char.get(id, '<UNK>') for id in token_ids])
    
    def tokenize(self, text):
        return list(text)
