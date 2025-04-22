import matplotlib.pyplot as plt
import seaborn as sns
from char_level_tokenization import CharacterTokenizer
from byte_tokenization import ByteTokenizer
from word_tokenization import WordTokenizer
from bpe_tokenization import BPETokenizer

with open("wiki_corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

min_freq = 1
vocab_size = 100

tokenizers = {
    "Character": CharacterTokenizer(),
    "Byte": ByteTokenizer(),
    "Word": WordTokenizer(min_freq=min_freq),
    "BPE": BPETokenizer(vocab_size=vocab_size)
}

for name, tokenizer in tokenizers.items():
    tokenizer.fit([text])

data = []
for name, tokenizer in tokenizers.items():
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text)
    reconstructed = tokenizer.decode(token_ids)
    reconstruction_success = reconstructed == text
    vocab = getattr(tokenizer, 'vocab_size', len(set(tokens)))
    data.append({
        'Tokenizer': name,
        'Vocab Size': vocab,
        'Token Count': len(tokens),
        'Tokens/Char': len(tokens)/len(text),
        'Reconstruction': reconstruction_success
    })

plt.figure(figsize=(8, 5))
sns.barplot(x=[d['Tokenizer'] for d in data], y=[d['Vocab Size'] for d in data], palette="viridis")
plt.title('Vocabulary Size by Tokenizer')
plt.ylabel('Vocabulary Size')
plt.xlabel('Tokenizer')
plt.tight_layout()
plt.savefig('vocab_size_by_tokenizer.png')
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(x=[d['Tokenizer'] for d in data], y=[d['Token Count'] for d in data], palette="magma")
plt.title('Token Count by Tokenizer')
plt.ylabel('Token Count')
plt.xlabel('Tokenizer')
plt.tight_layout()
plt.savefig('token_count_by_tokenizer.png')
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(x=[d['Tokenizer'] for d in data], y=[d['Tokens/Char'] for d in data], palette="cubehelix")
plt.title('Tokens per Character Ratio by Tokenizer')
plt.ylabel('Tokens per Character')
plt.xlabel('Tokenizer')
plt.tight_layout()
plt.savefig('tokens_per_char_by_tokenizer.png')
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(x=[d['Tokenizer'] for d in data], y=[int(d['Reconstruction']) for d in data], palette="Set2")
plt.title('Reconstruction Success by Tokenizer')
plt.ylabel('Reconstruction Success (1=Yes, 0=No)')
plt.xlabel('Tokenizer')
plt.ylim(0, 1.1)
plt.tight_layout()
plt.savefig('reconstruction_success_by_tokenizer.png')
plt.close()

print("Visualization images saved: vocab_size_by_tokenizer.png, token_count_by_tokenizer.png, tokens_per_char_by_tokenizer.png, reconstruction_success_by_tokenizer.png")
