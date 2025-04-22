
class ByteTokenizer:
    """
    A byte-level tokenizer that splits text into individual bytes.
    """
    
    def __init__(self):
        self.byte_to_id = {i: i for i in range(256)}
        self.id_to_byte = {i: i for i in range(256)}
        self.vocab_size = 256
        
        # self.special_tokens = {'<PAD>': 256, '<UNK>': 257}
        # for token, idx in self.special_tokens.items():
        #     self.byte_to_id[token] = idx
        #     self.id_to_byte[idx] = token
        #     self.vocab_size += 1
    
    def fit(self, texts=None):
        print(f"Vocabulary size: {self.vocab_size} bytes (fixed)")
        return self
    
    def encode(self, text):
        byte_values = list(text.encode('utf-8'))
        return byte_values
    
    def decode(self, byte_ids):
        try:
            bytes_data = bytes(byte_ids)
            return bytes_data.decode('utf-8')
        except UnicodeDecodeError:
            return bytes(byte_ids).decode('utf-8', errors='replace')
    
    def tokenize(self, text):

        byte_values = list(text.encode('utf-8'))
        return [f"0x{byte:02x}" for byte in byte_values]

