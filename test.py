from char_level_tokenization import CharacterTokenizer
from byte_tokenization import ByteTokenizer
from word_tokenization import WordTokenizer
from bpe_tokenization import BPETokenizer

def test_tokenizers(text):

    print(f"\n{'='*80}")
    print(f"Testing tokenization methods on: \"{text}\"")
    print(f"{'='*80}\n")

    tokenizers = {
        "Character": CharacterTokenizer(),
        "Byte": ByteTokenizer(),
        "Word": WordTokenizer(min_freq=1),
        "BPE": BPETokenizer(vocab_size=100)
    }
    
    print("Training tokenizers on input text...")
    for name, tokenizer in tokenizers.items():
        print(f"- Training {name} tokenizer...")
        tokenizer.fit([text])
    
    for name, tokenizer in tokenizers.items():
        print(f"\n{'-'*40}")
        print(f"{name} Tokenization")
        print(f"{'-'*40}")
        
        tokens = tokenizer.tokenize(text)
        
        with open(f"{name}_tokens.txt", "w", encoding="utf-8") as f:
            for token in tokens:
                f.write(str(token) + "\n")
        
        token_ids = tokenizer.encode(text)
        
        reconstructed = tokenizer.decode(token_ids)
        
        print(f"Vocabulary size: {getattr(tokenizer, 'vocab_size', 'N/A')}")
        print(f"Number of tokens: {len(tokens)}")
        print(f"Tokens/character ratio: {len(tokens)/len(text):.2f}")
        
        if len(tokens) > 20:
            print(f"Tokens (first 20 of {len(tokens)}): {tokens[:20]}...")
        else:
            print(f"Tokens ({len(tokens)}): {tokens}")
        
        if len(token_ids) > 10:
            print(f"Token IDs (first 10 of {len(token_ids)}): {token_ids[:10]}...")
        else:
            print(f"Token IDs ({len(token_ids)}): {token_ids}")
        
        reconstruction_success = "Success" if reconstructed == text else "Failed"
        print(f"Reconstruction: {reconstruction_success}")
        
        if reconstruction_success == "Failed":
            print(f"Original: {text}")
            print(f"Reconstructed: {reconstructed}")
    
    print(f"\n{'-'*40}")
    print("Summary Comparison")
    print(f"{'-'*40}")
    
    print(f"{'Tokenizer':<12} {'Vocab Size':<12} {'Token Count':<12} {'Tokens/Char':<12} {'Reconstruction':<12}")
    print(f"{'-'*60}")
    
    for name, tokenizer in tokenizers.items():
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text)
        reconstructed = tokenizer.decode(token_ids)
        reconstruction_success = "✓" if reconstructed == text else "✗"
        
        vocab_size = str(getattr(tokenizer, 'vocab_size', 'N/A'))
        token_count = str(len(tokens))
        tokens_per_char = f"{len(tokens)/len(text):.2f}"
        
        print(f"{name:<12} {vocab_size:<12} {token_count:<12} {tokens_per_char:<12} {reconstruction_success:<12}")

def main():
    try:
        with open("wiki_corpus.txt", "r", encoding="utf-8") as f:
            test_text = f.read()
        print("Using text from wiki_corpus.txt.")
    except Exception as e:
        print(f"Failed to read wiki_corpus.txt: {e}")
        test_text = "The quick brown fox jumps over the lazy dog. This sentence contains all letters in the English alphabet."
        print(f"Using default text: \"{test_text}\"")
    
    test_tokenizers(test_text)
    
    print("\nTest complete! You can modify this script to test with different parameters or texts.")

if __name__ == "__main__":
    main()