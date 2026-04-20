"""Evo 2 zero-shot variant effect prediction. Requires A100-80GB GPU via Modal."""

from __future__ import annotations

from typing import Any

import numpy as np

from genbench.registry import register_model


@register_model("evo2")
class Evo2Model:
    """40B DNA foundation model. >90% accuracy on BRCA1 SGE."""

    name = "evo2"

    def predict(self, inputs: dict[str, Any]) -> np.ndarray:
        reference = inputs["reference_sequence"]
        variants = inputs["variants"]  # list of (position, ref, alt)

        from genbench.app import app
        from genbench.config import MODELS_PATH
        from genbench.infra.images import gpu_image
        from genbench.infra.volumes import models_vol

        @app.function(
            image=gpu_image, gpu="A100-80GB", memory=131072,
            volumes={MODELS_PATH: models_vol}, timeout=7200,
        )
        def _predict_remote(ref_seq: str, variants: list, ctx: int = 8192) -> list[float]:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained("togethercomputer/evo-2-40b", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                "togethercomputer/evo-2-40b", trust_remote_code=True, torch_dtype=torch.bfloat16
            ).eval().cuda()

            scores = []
            for pos, ref, alt in variants:
                start = max(0, pos - ctx // 2)
                end = min(len(ref_seq), pos + ctx // 2)
                context = ref_seq[start:end]
                lp = pos - start

                ref_tok = tokenizer(context, return_tensors="pt").to("cuda")
                alt_seq = context[:lp] + alt + context[lp + len(ref):]
                alt_tok = tokenizer(alt_seq, return_tensors="pt").to("cuda")

                with torch.no_grad():
                    ref_logits = model(**ref_tok).logits
                    alt_logits = model(**alt_tok).logits

                ref_lp = torch.log_softmax(ref_logits[0, lp], dim=-1)
                alt_lp = torch.log_softmax(alt_logits[0, lp], dim=-1)

                r_tok = tokenizer.encode(ref, add_special_tokens=False)[0]
                a_tok = tokenizer.encode(alt, add_special_tokens=False)[0]
                scores.append(float(alt_lp[a_tok] - ref_lp[r_tok]))

            return scores

        return np.array(_predict_remote.remote(reference, variants))
