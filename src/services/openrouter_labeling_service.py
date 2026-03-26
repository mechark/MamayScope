from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from src.core.settings import settings
from src.prompts.system import LABELING_PROMPT

_DEFAULT_CHAT_COMPLETIONS_PATH = "chat/completions"


def _chat_completions_url(base_url: str) -> str:
    """
    Accept either a full chat-completions URL or an OpenAI-style API base (…/v1).
    """
    base = (base_url or "").strip().rstrip("/")
    if not base:
        base = "https://openrouter.ai/api/v1"
    if base.endswith("chat/completions"):
        return base
    return f"{base}/{_DEFAULT_CHAT_COMPLETIONS_PATH}"


def _csv_slugs(raw: str) -> list[str]:
    return [p.strip().lower() for p in raw.split(",") if p.strip()]


def _build_provider_routing(
    *,
    provider_only: list[str] | None,
    provider_order: list[str] | None,
    allow_fallbacks: bool,
) -> dict[str, Any] | None:
    only = [p.lower() for p in (provider_only or []) if p.strip()]
    order = [p.lower() for p in (provider_order or []) if p.strip()]
    if not only and not order:
        return None
    out: dict[str, Any] = {"allow_fallbacks": allow_fallbacks}
    if only:
        out["only"] = only
    if order:
        out["order"] = order
    return out


def _parse_openrouter_allowed_providers(response_text: str) -> list[str] | None:
    """OpenRouter returns JSON: error.metadata.available_providers for key/provider mismatch."""
    try:
        d = json.loads(response_text)
    except json.JSONDecodeError:
        return None
    err = d.get("error")
    if not isinstance(err, dict):
        return None
    msg = str(err.get("message", ""))
    if "No allowed providers" not in msg:
        return None
    meta = err.get("metadata")
    if not isinstance(meta, dict):
        return None
    raw = meta.get("available_providers")
    if not isinstance(raw, list) or not raw:
        return None
    return [str(x).strip().lower() for x in raw if str(x).strip()]


def _intersect_provider_allowlist(api_allowed: list[str]) -> list[str]:
    """If user set OPENROUTER_PROVIDER_ONLY, use that ∩ API list; else use API list."""
    user = set(_csv_slugs(settings.OPENROUTER_PROVIDER_ONLY))
    if not user:
        return api_allowed
    hit = [p for p in api_allowed if p in user]
    return hit if hit else api_allowed


def _extract_assistant_text(data: dict[str, Any]) -> str:
    """Normalize choices[0].message.content (str, None, or multimodal list)."""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    msg = choices[0].get("message")
    if not isinstance(msg, dict):
        return ""
    raw = msg.get("content")
    if raw is None:
        for key in ("text", "reasoning", "reasoning_content"):
            v = msg.get(key)
            if isinstance(v, str) and v.strip():
                return v
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for block in raw:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
        return "\n".join(parts)
    return str(raw)


def _build_request_provider_payload(base: dict[str, Any] | None) -> dict[str, Any]:
    """Merge routing + data_collection for OpenRouter provider object."""
    prov = dict(base) if base else {}
    dc = settings.OPENROUTER_PROVIDER_DATA_COLLECTION
    if dc in ("allow", "deny"):
        prov["data_collection"] = dc
    return prov


@dataclass(frozen=True)
class FeatureLabelResult:
    thought_process: str | None
    label: str
    raw_model_output: str


class OpenRouterLabelingService:
    """
    Minimal OpenRouter chat client for neuron/feature labeling.

    Note: Neuronpedia export format is not handled here; we only return parsed JSON fields.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        provider_only: list[str] | None = None,
        provider_order: list[str] | None = None,
        provider_allow_fallbacks: bool | None = None,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        retry_delay_s: float = 2.0,
    ):
        self.api_key = api_key or (settings.OPENROUTER_API_KEY or "").strip() or None
        if not self.api_key:
            raise RuntimeError(
                "Missing OpenRouter API key. Set OPENROUTER_API_KEY in your environment/.env "
                "or pass api_key=... to OpenRouterLabelingService."
            )

        self.model = model or (settings.OPENROUTER_MODEL or "").strip() or "meta-llama/llama-3.3-70b-instruct"
        resolved_base = (base_url or settings.OPENROUTER_BASE_URL or "").strip() or "https://openrouter.ai/api/v1"
        self.chat_completions_url = _chat_completions_url(resolved_base)
        self._headers: dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "MamayScope/openrouter-labeling (httpx)",
        }
        ref = (settings.OPENROUTER_HTTP_REFERER or "").strip()
        if ref:
            self._headers["HTTP-Referer"] = ref
        title = (settings.OPENROUTER_APP_TITLE or "").strip()
        if title:
            self._headers["X-Title"] = title
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_delay_s = retry_delay_s

        allow_fb = (
            settings.OPENROUTER_PROVIDER_ALLOW_FALLBACKS
            if provider_allow_fallbacks is None
            else provider_allow_fallbacks
        )
        only_src = (
            provider_only
            if provider_only is not None
            else _csv_slugs(settings.OPENROUTER_PROVIDER_ONLY)
        )
        order_src = (
            provider_order
            if provider_order is not None
            else _csv_slugs(settings.OPENROUTER_PROVIDER_ORDER)
        )
        self._provider = _build_provider_routing(
            provider_only=only_src if only_src else None,
            provider_order=order_src if order_src else None,
            allow_fallbacks=bool(allow_fb),
        )

    @staticmethod
    def _extract_first_json_object(text: str) -> dict[str, Any]:
        # If model already returned pure JSON, parse directly.
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # Otherwise, extract the first {...} block.
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise ValueError("Could not find JSON object in model output.")

        candidate = m.group(0)
        parsed = json.loads(candidate)
        if not isinstance(parsed, dict):
            raise ValueError("Parsed JSON is not an object.")
        return parsed

    def _build_prompts(
        self,
        *,
        feature_id: int,
        contexts: list[str],
        avoid_labels: list[str] | None = None,
        force_specific: bool = False,
    ) -> tuple[str, str]:
        system_prompt = LABELING_PROMPT
        extra_rules = ""
        if force_specific:
            extra_rules += (
                "\nAdditional constraints:\n"
                "- Avoid generic labels like 'Key Concept' or 'Semantic Element'.\n"
                "- Label should name a concrete pattern (topic, lexical family, construct, role).\n"
                "- If uncertain, still pick the most specific plausible pattern from contexts.\n"
            )
        if avoid_labels:
            banned = ", ".join(sorted({x.strip() for x in avoid_labels if str(x).strip()}))
            if banned:
                extra_rules += f"\nDo NOT use these labels: {banned}\n"

        user_prompt = f"Feature #{feature_id} Activations:\n\n" + "\n\n".join(contexts) + extra_rules
        return system_prompt, user_prompt

    def label_feature(
        self,
        *,
        feature_id: int,
        contexts: list[str],
        avoid_labels: list[str] | None = None,
        force_specific: bool = False,
    ) -> FeatureLabelResult:
        if not contexts:
            raise ValueError("contexts must be non-empty")

        system_prompt, user_prompt = self._build_prompts(
            feature_id=feature_id,
            contexts=contexts,
            avoid_labels=avoid_labels,
            force_specific=force_specific,
        )

        def _post_json(body: dict[str, Any]) -> httpx.Response:
            with httpx.Client(timeout=self.timeout_s, follow_redirects=True) as client:
                return client.post(
                    self.chat_completions_url,
                    headers=self._headers,
                    json=body,
                )

        def _http_error_detail(resp: httpx.Response) -> str:
            text = (resp.text or "").strip()
            if text.startswith("<!DOCTYPE") or text.startswith("<html"):
                return (
                    "response looks like HTML (wrong URL or blocked). "
                    f"POST URL was {self.chat_completions_url!r}. "
                    "Set OPENROUTER_BASE_URL=https://openrouter.ai/api/v1 in .env "
                    "(do not use https://openrouter.ai/v1/..., that serves the website)."
                )
            snippet = text[:1200] if text else "(empty body)"
            if resp.status_code == 404 and "No allowed providers" in snippet:
                snippet += (
                    " | Auto-fix: on this error the client will retry with provider.only from "
                    "`available_providers` and OPENROUTER_FALLBACK_MODEL if needed. "
                    "You can also set OPENROUTER_PROVIDER_ONLY / OPENROUTER_MODEL in .env."
                )
            if resp.status_code == 404 and (
                "guardrail" in snippet.lower()
                or "data policy" in snippet.lower()
                or "privacy" in snippet.lower()
            ):
                snippet += (
                    " | OpenRouter account privacy is blocking all endpoints: open "
                    "https://openrouter.ai/settings/privacy and allow providers that match your model, "
                    "or set OPENROUTER_PROVIDER_DATA_COLLECTION=allow (default in MamayScope) if you "
                    "previously set deny at the account level."
                )
            return snippet

        def _one_completion_round(payload_base: dict[str, Any]) -> tuple[dict[str, Any] | None, httpx.Response | None]:
            """Returns (data, None) on success, (None, error_response) on final HTTP error."""
            for attempt in range(self.max_retries):
                try:
                    for use_response_format in (True, False):
                        payload = dict(payload_base)
                        if use_response_format:
                            payload["response_format"] = {"type": "json_object"}
                        r = _post_json(payload)
                        if r.status_code in (400, 404, 422) and use_response_format:
                            continue
                        if r.status_code >= 400:
                            return None, r
                        try:
                            data = r.json()
                        except json.JSONDecodeError as exc:
                            detail = _http_error_detail(r)
                            raise ValueError(
                                f"Expected JSON from OpenRouter but got non-JSON body ({r.status_code}): {detail}"
                            ) from exc
                        return data, None
                    return None, r
                except httpx.HTTPError:
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay_s * (2**attempt))
                        continue
                    raise
            raise RuntimeError("OpenRouter request failed after retries (no response).")

        # Re-resolve provider routing after settings may be intersected with API allowlist.
        allow_fb = settings.OPENROUTER_PROVIDER_ALLOW_FALLBACKS

        for route_fix in range(4):
            payload_base: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 250,
            }
            payload_base["provider"] = _build_request_provider_payload(self._provider)

            data, err_resp = _one_completion_round(payload_base)
            if data is not None:
                content = _extract_assistant_text(data).strip()
                if not content:
                    raise ValueError(
                        f"Empty assistant text from model (content was null or non-text). Raw: {data!r}"
                    )

                obj = self._extract_first_json_object(content)
                thought_process = obj.get("thought_process")
                label = obj.get("label")
                if not isinstance(label, str) or not label.strip():
                    raise ValueError(f"Missing/invalid `label` in parsed JSON: {obj!r}")
                return FeatureLabelResult(
                    thought_process=str(thought_process) if thought_process is not None else None,
                    label=label.strip(),
                    raw_model_output=content,
                )

            assert err_resp is not None
            body = (err_resp.text or "").strip()
            body_l = body.lower()
            privacy_404 = err_resp.status_code == 404 and (
                "guardrail" in body_l
                or "data policy" in body_l
                or "no endpoints available matching" in body_l
            )
            if err_resp.status_code == 404:
                if privacy_404 and self._provider is not None and route_fix < 3:
                    # Strict provider.only/order + account privacy can leave zero endpoints.
                    self._provider = None
                    continue
                api_list = _parse_openrouter_allowed_providers(body)
                if api_list and route_fix < 2:
                    use_only = _intersect_provider_allowlist(api_list)
                    order_src = _csv_slugs(settings.OPENROUTER_PROVIDER_ORDER)
                    if order_src:
                        allow_set = set(use_only)
                        order_src = [p for p in order_src if p in allow_set]
                    self._provider = _build_provider_routing(
                        provider_only=use_only,
                        provider_order=order_src if order_src else None,
                        allow_fallbacks=bool(allow_fb),
                    )
                    fb = (settings.OPENROUTER_FALLBACK_MODEL or "").strip()
                    if fb and self.model != fb:
                        self.model = fb
                        continue
                    continue

            detail = _http_error_detail(err_resp)
            raise RuntimeError(
                f"OpenRouter labeling failed for feature #{feature_id} "
                f"(url={self.chat_completions_url!r} model={self.model!r}): "
                f"{err_resp.status_code} {err_resp.reason_phrase}: {detail}"
            )

        raise RuntimeError(
            f"OpenRouter labeling failed for feature #{feature_id}: exhausted provider/model auto-retries"
        )

