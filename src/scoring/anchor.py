"""Offline calibration anchor: the P4 fine-tuned judge, loaded from the HF Hub.

Requires the `gpu` extra + a CUDA GPU. Runs locally only -- never CI or EC2.
"""

from __future__ import annotations

import json
import re
from typing import Any, cast

from judgekit import Judgment, Sample

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "umairrasheed828/calibrated-research-qa-judge"

# NOTE: for exact parity, align this with your P4 src/judge/prompt.py format.
SYSTEM_PROMPT = (
    "You are a strict evaluator of a research answer. Score two axes, each a whole "
    "integer 1-5.\n"
    "faithfulness: is every claim in the ANSWER grounded in the CONTEXT? "
    "5=fully grounded; 3=one minor unsupported detail; 1=fabricated. Judge ONLY grounding.\n"
    "relevance: does the ANSWER address the QUESTION? 5=fully; 3=partially; 1=off-topic. "
    "Judge ONLY topical fit.\n"
    'Return ONLY JSON: {"faithfulness": N, "relevance": N}'
)


def _user_turn(sample: Sample) -> str:
    return (
        f"QUESTION:\n{sample.input}\n\n"
        f"CONTEXT:\n{sample.context or ''}\n\n"
        f"ANSWER:\n{sample.output}"
    )


def _parse(text: str) -> dict[str, int]:
    try:
        start, end = text.find("{"), text.rfind("}")
        data = json.loads(text[start : end + 1])
        return {
            "faithfulness": int(data["faithfulness"]),
            "relevance": int(data["relevance"]),
        }
    except Exception:
        found = dict(re.findall(r'"(faithfulness|relevance)":\s*(\d)', text))
        return {
            "faithfulness": int(found.get("faithfulness", 3)),
            "relevance": int(found.get("relevance", 3)),
        }


class FineTunedJudge:
    """The P4 QLoRA judge as a judgekit-compatible Judge (structural typing)."""

    def __init__(self) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=bnb, device_map="auto"
        )
        self.model = PeftModel.from_pretrained(base, ADAPTER)
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def score(self, sample: Sample) -> Judgment:
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_turn(sample)},
        ]
        inputs = cast(
            Any,
            self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ),
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=32, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1] :]
        text = cast(str, self.tokenizer.decode(gen, skip_special_tokens=True))
        return Judgment(scores=_parse(text), rationale="")
