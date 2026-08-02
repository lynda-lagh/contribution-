"""
Logit-level scoring -- the mechanism behind Chapter 1's LogitParser.

We never generate. We read P("Yes") vs P("No") at the first response-token
position, which removes output format from the measurement entirely: the model
cannot be penalised for refusing, hedging, rambling, or echoing pretraining text.

This is what replaces KG-LLM's substring matching, whose lenient variant fires
on "k[no]w" / "can[no]t" / "a[no]ther".
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _first_token_ids(tokenizer, words: list[str]) -> dict[str, list[int]]:
    """
    First-token id for each answer word, in the positions the model will actually
    see. Most tokenizers treat "Yes" and " Yes" differently, so we collect both.
    """
    out: dict[str, list[int]] = {}
    for w in words:
        ids = set()
        for variant in (w, " " + w, w.lower(), " " + w.lower()):
            enc = tokenizer(variant, add_special_tokens=False)["input_ids"]
            if enc:
                ids.add(enc[0])
        out[w] = sorted(ids)
    return out


@torch.no_grad()
def yes_no_probabilities(
    model,
    tokenizer,
    prompts: list[str],
    device: str = "cuda",
    batch_size: int = 8,
) -> list[tuple[float, float]]:
    """
    For each prompt (already wrapped in the Alpaca template and ending at
    "### Response:\n"), return (P_yes, P_no) renormalised over the two options.

    Renormalising over {Yes, No} is deliberate: we are asking which of the two the
    model prefers, not how much mass it puts on answering at all.
    """
    model.eval()
    ids = _first_token_ids(tokenizer, ["Yes", "No"])
    yes_ids, no_ids = ids["Yes"], ids["No"]

    out: list[tuple[float, float]] = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=512).to(device)
        logits = model(**enc).logits              # (B, T, V)

        # ★ The prediction position depends on WHICH SIDE was padded.
        #
        #   right padding: [tok tok tok PAD PAD] -> last real token is at
        #                  attention_mask.sum() - 1
        #   left  padding: [PAD PAD tok tok tok] -> last real token is at T-1,
        #                  and attention_mask.sum()-1 lands INSIDE THE PAD RUN
        #
        # Chapter 1 sets `padding_side = "left"` (correct for next-token scoring),
        # so using the right-padding formula reads a padded position. Those
        # positions attend to nothing, which in fp16 yields NaN logits -- the whole
        # row of probabilities becomes NaN and the result is silently meaningless.
        if getattr(tokenizer, "padding_side", "right") == "left":
            next_logits = logits[:, -1, :]                       # (B, V)
        else:
            last_idx = enc["attention_mask"].sum(dim=1) - 1
            rows = torch.arange(logits.size(0), device=logits.device)
            next_logits = logits[rows, last_idx, :]              # (B, V)

        if not torch.isfinite(next_logits).all():
            raise RuntimeError(
                "Non-finite logits at the scoring position. Usually a padded "
                "position was read (check tokenizer.padding_side) or fp16 overflow."
            )

        probs = F.softmax(next_logits.float(), dim=-1)
        p_yes = probs[:, yes_ids].sum(dim=-1)
        p_no = probs[:, no_ids].sum(dim=-1)
        denom = (p_yes + p_no).clamp_min(1e-12)
        out.extend(zip((p_yes / denom).tolist(), (p_no / denom).tolist()))
    return out


@torch.no_grad()
def constrained_choice(
    model,
    tokenizer,
    prompts: list[str],
    options: tuple[str, str] = ("Yes, this is true.", "No, this is not true."),
    device: str = "cuda",
    batch_size: int = 4,
) -> list[str]:
    """
    Belief under a FORCED format: score each full answer string by its
    length-normalised log-likelihood and return the higher one.

    Sits between LogitParser (no format) and StrictParser (full format), so it
    isolates the cost of producing a well-formed SEQUENCE rather than a token.
    """
    model.eval()
    chosen: list[str] = []
    for i in range(0, len(prompts), batch_size):
        for prompt in prompts[i : i + batch_size]:
            scores = []
            for opt in options:
                full = tokenizer(prompt + opt, return_tensors="pt",
                                 truncation=True, max_length=512).to(device)
                n_prompt = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
                labels = full["input_ids"].clone()
                labels[:, :n_prompt] = -100                    # score the answer only
                loss = model(**full, labels=labels).loss       # mean NLL over answer
                scores.append(-loss.item())                    # higher = more likely
            chosen.append(options[int(scores[1] > scores[0])])
    return chosen


@torch.no_grad()
def sequence_logprob(model, tokenizer, prompt: str, answer: str,
                     device: str = "cuda") -> float:
    """Length-normalised log-probability of `answer` -- one of Chapter 4's
    three confidence sources (with P(True) and sampling disagreement)."""
    full = tokenizer(prompt + answer, return_tensors="pt",
                     truncation=True, max_length=512).to(device)
    n_prompt = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    labels = full["input_ids"].clone()
    labels[:, :n_prompt] = -100
    return -model(**full, labels=labels).loss.item()
