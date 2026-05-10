import torch
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER

VOCAB_PATH = r"D:\python\lib\site-packages\rwkv\rwkv_vocab_v20230424.txt"

EVAL_TEXTS = [
    "The future of artificial intelligence will depend on how we design",
    "Deep learning applications in healthcare include medical image analysis",
    "Natural language processing has evolved significantly with transformer models",
    "Quantum computing promises to revolutionize cryptography and optimization",
    "Climate change requires immediate action from governments and corporations",
    "The history of mathematics stretches back thousands of years",
    "Modern economics relies heavily on statistical models and data",
    "Space exploration has led to numerous technological innovations",
    "The human brain contains approximately eighty six billion neurons",
    "Renewable energy sources like solar and wind are becoming cost competitive",
    "Machine learning algorithms can now generate realistic images and text",
    "The theory of evolution explains the diversity of life",
    "Blockchain technology enables decentralized trust and transparency",
    "Understanding consciousness remains a challenge in neuroscience",
    "Global supply chains have been disrupted by geopolitical events",
    "The development of agriculture marked a fundamental shift in civilization",
    "Particle physics explores the fundamental constituents of matter",
    "Cybersecurity threats continue to evolve as technology becomes more integrated",
]

_tok = None

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def init_tokenizer(vocab_path=None):
    global _tok
    path = vocab_path or VOCAB_PATH
    if _tok is None:
        _tok = TRIE_TOKENIZER(path)
    return _tok

def load_eval_texts():
    return EVAL_TEXTS