import torch
from .model import rwkv7_fwd_inject, _dev


@torch.no_grad()
def generate(prefix, w, nL, C, H, N, inject=None, max_new=60, temperature=0.7, top_k=40):
    """Generate text continuation with optional τ injection.
    inject: dict as in rwkv7_fwd_inject."""
    dev = _dev()
    ids = list(tok_encode(prefix))
    for _ in range(max_new):
        logits = rwkv7_fwd_inject(ids, w, nL, C, H, N, inject=inject)
        logits_last = logits[-1].float()
        if temperature > 0:
            logits_last = logits_last / temperature
            v, _ = torch.topk(logits_last, top_k)
            logits_last[logits_last < v[-1]] = -float("inf")
            probs = torch.softmax(logits_last, dim=-1)
            ids.append(torch.multinomial(probs, 1).item())
        else:
            ids.append(torch.argmax(logits_last).item())
    prefix_ids = tok_encode(prefix)
    return tok_decode(ids[len(prefix_ids):])


def generate_compare(prefix, configs, w, nL, C, H, N, max_new=50, temperature=0.7, top_k=40):
    """Generate under multiple injection configs side-by-side.
    configs: list of (tag, inject_dict_or_None).
    Returns dict {tag: generated_text}."""
    results = {}
    for tag, inject in configs:
        results[tag] = generate(prefix, w, nL, C, H, N, inject=inject,
                                max_new=max_new, temperature=temperature, top_k=top_k)
    return results


_tok = None

def tok_encode(text):
    global _tok
    if _tok is None:
        from .utils import init_tokenizer
        _tok = init_tokenizer()
    return _tok.encode(text)

def tok_decode(ids):
    global _tok
    if _tok is None:
        from .utils import init_tokenizer
        _tok = init_tokenizer()
    return _tok.decode(ids)