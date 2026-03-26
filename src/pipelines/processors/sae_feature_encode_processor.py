import logging
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from sae_lens import SAE
from transformers import AutoTokenizer

from src.core.settings import settings
from src.pipelines.base import PipelineStep


def resolve_sae_device(preference: str) -> str:
    p = preference.lower().strip()
    if p == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return p


class SaeFeatureEncodeProcessor(PipelineStep):
    """Runs residual-stream activations through a pretrained SAELens SAE and collects sparse feature indices."""

    def __init__(
        self,
        repo_id: str | None = None,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        device_pref: str | None = None,
        target_layer: int | None = None,
    ):
        self.logger = logging.getLogger(__name__)
        self.repo_id = repo_id or settings.SAE_HF_REPO_ID
        self.revision = revision or settings.SAE_HF_REVISION
        self.cache_dir = Path(cache_dir or settings.SAE_SNAPSHOT_CACHE_DIR)
        self.device_pref = device_pref or settings.SAE_DEVICE
        self.target_layer = target_layer if target_layer is not None else settings.TARGET_LAYER

        self._sae: SAE | None = None
        self._device: str | None = None
        self._snapshot_path: Path | None = None
        self._tokenizer: AutoTokenizer | None = None

    def _snapshot_local_dir(self) -> Path:
        safe = self.repo_id.replace("/", "__")
        return self.cache_dir / f"{safe}_{self.revision[:12]}"

    def _ensure_sae(self) -> None:
        if self._sae is not None:
            return

        local_dir = self._snapshot_local_dir()
        local_dir.parent.mkdir(parents=True, exist_ok=True)

        if not (local_dir / "cfg.json").is_file():
            self.logger.info(
                "Downloading SAE snapshot %s @ %s → %s",
                self.repo_id,
                self.revision,
                local_dir,
            )
            snapshot_download(
                repo_id=self.repo_id,
                revision=self.revision,
                local_dir=str(local_dir),
            )
        else:
            self.logger.info("Using cached SAE snapshot at %s", local_dir)

        self._snapshot_path = local_dir
        device = resolve_sae_device(self.device_pref)
        self._device = device

        try:
            self._sae = SAE.load_from_disk(str(local_dir), device=device)
        except Exception as e:
            if device == "mps":
                self.logger.warning("SAE.load_from_disk on mps failed (%s); retrying cpu", e)
                self._device = "cpu"
                self._sae = SAE.load_from_disk(str(local_dir), device="cpu")
            else:
                raise

        self._sae.eval()
        self.logger.info("Loaded SAE from %s on %s", local_dir, self._device)

    def _metadata_model_name(self) -> str | None:
        meta = getattr(self._sae.cfg, "metadata", None) if self._sae else None
        if meta is None:
            return None
        mn = getattr(meta, "model_name", None)
        if mn:
            return str(mn)
        if isinstance(meta, dict):
            v = meta.get("model_name")
            return str(v) if v else None
        to_dict = getattr(meta, "to_dict", None)
        if callable(to_dict):
            d = to_dict()
            v = d.get("model_name") if isinstance(d, dict) else None
            return str(v) if v else None
        return None

    def _resolved_model_name(self) -> str:
        from_env = (settings.NEURON_LABEL_MODEL_NAME or "").strip()
        if from_env:
            return from_env
        return (self._metadata_model_name() or "").strip()

    def _resolved_sae_id(self) -> str:
        from_env = (settings.NEURON_LABEL_SAE_ID or "").strip()
        if from_env:
            return from_env
        stem = self.repo_id.split("/")[-1]
        short = self.revision[:7] if len(self.revision) >= 7 else self.revision
        return f"layer_{self.target_layer}_{stem}_{short}"

    def _ensure_tokenizer(self) -> None:
        if self._tokenizer is not None:
            return
        model_name = self._resolved_model_name()
        if not model_name:
            return
        self.logger.info("Loading AutoTokenizer for model=%s", model_name)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_default_system_prompt=False,
            )
        except TypeError:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    @staticmethod
    def _flat_input_ids(raw: object) -> list[int]:
        if isinstance(raw, torch.Tensor):
            return [int(x) for x in raw.view(-1).tolist()]
        if isinstance(raw, (list, tuple)):
            if len(raw) == 0:
                return []
            if isinstance(raw[0], (list, tuple)):
                inner = raw[0]
                return [int(x) for x in inner]
            return [int(x) for x in raw]
        raise TypeError(f"Unexpected input_ids type: {type(raw)}")

    @staticmethod
    def _dedupe_id_lists(rows: list[list[int]]) -> list[list[int]]:
        seen: set[tuple[int, ...]] = set()
        out: list[list[int]] = []
        for r in rows:
            t = tuple(r)
            if t not in seen:
                seen.add(t)
                out.append(r)
        return out

    def _token_ids_for_text(self, text: str, seq_len: int) -> list[int]:
        self._ensure_tokenizer()
        if self._tokenizer is None:
            return [-1] * seq_len

        tok = self._tokenizer
        chat_candidates: list[list[int]] = []
        plain_candidates: list[list[int]] = []

        # MamayLM-Gemma-2-9B-IT (vLLM): chat template + tokenizer(..., add_special_tokens=False).
        # Mamay `/activations` often returns seq_len = prompt tokens + generated continuation; exact
        # length match to prompt-only encoding is uncommon. See model card:
        # https://huggingface.co/INSAIT-Institute/MamayLM-Gemma-2-9B-IT-v0.1
        if getattr(tok, "chat_template", None):
            messages = [{"role": "user", "content": text}]
            for add_generation_prompt in (True, False):
                try:
                    formatted = tok.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=add_generation_prompt,
                    )
                    enc = tok(
                        formatted,
                        add_special_tokens=False,
                        return_attention_mask=False,
                    )
                    chat_candidates.append(self._flat_input_ids(enc["input_ids"]))
                except Exception:
                    pass
                try:
                    templated = tok.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=add_generation_prompt,
                    )
                    chat_candidates.append(self._flat_input_ids(templated))
                except Exception:
                    pass

        for add_special in (False, True):
            enc = tok(text, add_special_tokens=add_special, return_attention_mask=False)
            plain_candidates.append(self._flat_input_ids(enc["input_ids"]))

        chat_candidates = self._dedupe_id_lists(chat_candidates)
        plain_candidates = self._dedupe_id_lists(plain_candidates)
        combined = chat_candidates + plain_candidates

        for ids in combined:
            if len(ids) == seq_len:
                return ids

        bos_id = getattr(tok, "bos_token_id", None)
        for ids in combined:
            if bos_id is not None and len(ids) == seq_len + 1 and ids[0] == bos_id:
                return ids[1:]

        # Prefer longest chat prompt that fits in seq_len, then pad — covers prompt + generated tail.
        chat_fit = [x for x in chat_candidates if len(x) <= seq_len]
        if chat_fit:
            best = max(chat_fit, key=len)
            tail = seq_len - len(best)
            if tail > 0:
                self.logger.debug(
                    "Prompt tokens=%s activation seq_len=%s (%s positions likely model-generated after prompt); text=%r",
                    len(best),
                    seq_len,
                    tail,
                    text[:80],
                )
            return best + [-1] * tail

        plain_fit = [x for x in plain_candidates if len(x) <= seq_len]
        if plain_fit:
            best = max(plain_fit, key=len)
            tail = seq_len - len(best)
            if tail > 0:
                self.logger.warning(
                    "No chat prompt fit seq_len=%s; padding from plain encode (len=%s, +%s tail); text=%r",
                    seq_len,
                    len(best),
                    tail,
                    text[:80],
                )
            return best + [-1] * tail

        # Shorter seq on server than our shortest chat prompt: prefix-truncate shortest chat sequence.
        chat_over = [x for x in chat_candidates if len(x) >= seq_len]
        if chat_over:
            best = min(chat_over, key=len)
            self.logger.warning(
                "Truncating chat token ids %s → %s (server seq shorter than prompt template); text=%r",
                len(best),
                seq_len,
                text[:80],
            )
            return best[:seq_len]

        plain_over = [x for x in plain_candidates if len(x) >= seq_len]
        if plain_over:
            best = min(plain_over, key=len)
            self.logger.warning(
                "Truncating plain token ids %s → %s (text=%r); verify NEURON_LABEL_MODEL_NAME matches Mamay",
                len(best),
                seq_len,
                text[:80],
            )
            return best[:seq_len]

        return [-1] * seq_len

    def _token_ids_for_row(self, text: str, seq_len: int, input_ids: list[int] | None) -> list[int]:
        if input_ids is not None:
            ids = [int(x) for x in input_ids]
            if len(ids) == seq_len:
                return ids
            self.logger.warning(
                "Server provided input_ids len=%s but activation seq_len=%s; falling back to local tokenization (text=%r)",
                len(ids),
                seq_len,
                text[:80],
            )
        return self._token_ids_for_text(text, seq_len)

    def _tokens_field(
        self,
        text: str,
        fired_per_token: list[list[int]],
        input_ids: list[int] | None,
    ) -> list[dict]:
        seq_len = len(fired_per_token)
        ids = self._token_ids_for_row(text, seq_len, input_ids)
        tok = self._tokenizer
        out: list[dict] = []
        for pos in range(seq_len):
            tid = int(ids[pos]) if pos < len(ids) else -1
            if tok is not None and tid >= 0:
                token_str = tok.convert_ids_to_tokens(tid)
                if isinstance(token_str, bytes):
                    token_str = token_str.decode("utf-8", errors="replace")
                if token_str.startswith("▁"):
                    token_str = token_str[1:]
            else:
                token_str = "<unk>"
            out.append(
                {
                    "token_str": token_str,
                    "token_id": tid,
                    "fired_features": fired_per_token[pos],
                }
            )
        return out

    def _encode_row(self, acts: torch.Tensor) -> list[list[int]]:
        device = self._device or "cpu"
        if acts.numel() == 0:
            return []

        x = acts.unsqueeze(0).float().to(device)
        try:
            with torch.no_grad():
                feature_acts = self._sae.encode(x)
        except Exception as e:
            if device == "mps":
                self.logger.warning(
                    "sae.encode on mps failed (%s); moving SAE to cpu for this batch and onward",
                    e,
                )
                self._sae = self._sae.cpu()
                self._device = "cpu"
                x = x.cpu()
                with torch.no_grad():
                    feature_acts = self._sae.encode(x)
            else:
                raise

        if feature_acts.is_sparse:
            feature_acts = feature_acts.to_dense()
        feature_acts = feature_acts.squeeze(0)

        out: list[list[int]] = []
        for pos in range(feature_acts.shape[0]):
            row = feature_acts[pos]
            idx = (row > 0).nonzero(as_tuple=True)[0].cpu().tolist()
            out.append([int(i) for i in idx])
        return out

    async def run(self, data: dict) -> dict:
        output_tensors: list[torch.Tensor] = data.get("output_tensors", [])
        texts: list[str] = data.get("texts", [])
        labels = data.get("labels")
        input_ids_rows: list[list[int] | None] = data.get("input_ids", [])

        if not output_tensors:
            self.logger.warning("SaeFeatureEncodeProcessor: no output tensors")
            return {
                "done": data.get("done", False),
                "neuron_label_records": [],
            }

        self._ensure_sae()

        resolved_mn = self._resolved_model_name()
        model_for_json = resolved_mn if resolved_mn else "unknown"
        sae_id = self._resolved_sae_id()
        if not resolved_mn:
            self.logger.warning(
                "Set NEURON_LABEL_MODEL_NAME to your Mamay HF model id for real token_str/token_id; "
                "using placeholders until then."
            )

        if labels is None:
            labels_list: list = [None] * len(output_tensors)
        else:
            labels_list = list(labels)
            if len(labels_list) != len(output_tensors):
                self.logger.warning(
                    "labels length %s != tensors %s; padding with null",
                    len(labels_list),
                    len(output_tensors),
                )
                labels_list = (labels_list + [None] * len(output_tensors))[: len(output_tensors)]

        if len(texts) != len(output_tensors):
            self.logger.warning(
                "texts length %s != tensors %s; text field may mismatch",
                len(texts),
                len(output_tensors),
            )

        records: list[dict] = []
        for i, acts in enumerate(output_tensors):
            text = texts[i] if i < len(texts) else ""
            label = labels_list[i]
            row_input_ids = input_ids_rows[i] if i < len(input_ids_rows) else None
            per_token = self._encode_row(acts)
            rec = {
                "text": text,
                "label": label,
                "target_layer": self.target_layer,
                "model": model_for_json,
                "sae_id": sae_id,
                "tokens": self._tokens_field(text, per_token, row_input_ids),
            }
            records.append(rec)

        return {
            "done": data.get("done", False),
            "neuron_label_records": records,
        }
