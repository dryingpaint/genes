"""SaProt zero-shot variant effect prediction. Requires GPU via Modal."""

from __future__ import annotations

from typing import Any

import numpy as np

from genbench.registry import register_model


@register_model("saprot")
class SaProtModel:
    """Structure-aware protein LM. SOTA on ProteinGym DMS (Spearman ~0.48)."""

    name = "saprot"

    def predict(self, inputs: dict[str, Any]) -> np.ndarray:
        sequence = inputs["sequence"]
        variants = inputs["variants"]  # list of "A1T" style mutation strings

        # Import Modal function lazily to avoid GPU image issues on CPU
        from genbench.app import app
        from genbench.config import MODELS_PATH
        from genbench.infra.images import gpu_image
        from genbench.infra.volumes import models_vol

        @app.function(image=gpu_image, gpu="A10G", volumes={MODELS_PATH: models_vol}, timeout=3600)
        def _predict_remote(seq: str, variants: list[str]) -> list[float]:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer

            model_name = "westlake-repl/SaProt_650M_AF2"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForMaskedLM.from_pretrained(model_name)
            model.eval().cuda()

            scores = []
            for mut in variants:
                wt_aa, mut_aa = mut[0], mut[-1]
                pos = int(mut[1:-1])

                masked = list(seq)
                masked[pos - 1] = tokenizer.mask_token
                tok = tokenizer("".join(masked), return_tensors="pt").to("cuda")

                with torch.no_grad():
                    logits = model(**tok).logits

                mask_idx = (tok["input_ids"] == tokenizer.mask_token_id).nonzero()[0, 1]
                log_probs = torch.log_softmax(logits[0, mask_idx], dim=-1)

                wt_tok = tokenizer.encode(wt_aa, add_special_tokens=False)[0]
                mut_tok = tokenizer.encode(mut_aa, add_special_tokens=False)[0]
                scores.append(float(log_probs[mut_tok] - log_probs[wt_tok]))

            return scores

        return np.array(_predict_remote.remote(sequence, variants))
