"""A deterministic fake LLM for tests, CI, and offline eval runs.

It is NOT a language model. It inspects a machine-readable ``TASK:`` tag that the
agents place as the first line of the user message and returns a structured,
deterministic response derived purely from the provided evidence. This lets the
whole agent graph run end-to-end — and lets evals measure grounding/refusal —
without any network access or AWS credentials.

Supported tasks:

* ``TASK: decompose``  -> up to N sub-queries, one per line.
* ``TASK: summarize``  -> a one-line summary that quotes evidence and carries
  the chunk IDs it drew from.
* ``TASK: respond``    -> a final grounded answer with ``[doc_id:section]``
  citations, or an explicit insufficient-evidence refusal.

The response format mirrors what a well-prompted real model would return, so the
same parsing code works for both backends.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from cra.llm.base import BaseLLM, LLMResponse, Message, estimate_tokens

_TASK_RE = re.compile(r"^TASK:\s*(\w+)", re.IGNORECASE)
# Evidence lines look like: "[chunk_id] text..."
_EVIDENCE_RE = re.compile(r"^\[(?P<cid>[^\]]+)\]\s*(?P<text>.*)$")
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "are",
    "was",
    "were",
    "what",
    "which",
    "whose",
    "whom",
    "compare",
    "between",
    "their",
    "these",
    "those",
    "does",
    "did",
    "how",
    "why",
    "when",
    "where",
    "versus",
    "vs",
    "two",
    "both",
    "each",
    "per",
    "into",
    "about",
    "over",
    "under",
    "of",
    "in",
    "on",
    "to",
    "a",
    "an",
    "is",
    "it",
    "its",
    "as",
    "at",
    "by",
    "or",
}


@dataclass
class _Evidence:
    chunk_id: str
    text: str


def _parse_evidence(user_content: str) -> list[_Evidence]:
    out: list[_Evidence] = []
    for line in user_content.splitlines():
        m = _EVIDENCE_RE.match(line.strip())
        if m:
            out.append(_Evidence(m.group("cid"), m.group("text").strip()))
    return out


def _keywords(text: str) -> list[str]:
    toks = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in toks if len(t) > 2 and t not in _STOPWORDS]


@dataclass
class FakeLLM(BaseLLM):
    """Deterministic, evidence-grounded stand-in for a real chat model."""

    max_sub_queries: int = 3

    def _generate(self, messages: list[Message]) -> LLMResponse:
        user = next((m for m in reversed(messages) if m.role == "user"), None)
        content = user.content if user else ""
        task_match = _TASK_RE.search(content)
        task = task_match.group(1).lower() if task_match else "respond"

        if task == "decompose":
            text = self._decompose(content)
        elif task == "summarize":
            text = self._summarize(content)
        else:
            text = self._respond(content)

        prompt_text = "\n".join(m.content for m in messages)
        return LLMResponse(
            text=text,
            input_tokens=estimate_tokens(prompt_text),
            output_tokens=estimate_tokens(text),
        )

    # ---- task handlers ---------------------------------------------------
    def _decompose(self, content: str) -> str:
        """Produce up to N sub-queries from the question.

        The question is passed after a ``QUESTION:`` marker. We split on
        conjunctions ("and", "compare ... and ...") to approximate decomposition
        deterministically; if nothing splits, we return the question itself.
        """
        q = self._extract(content, "QUESTION:")
        parts = re.split(r"\b(?:and|,|;|as well as|along with)\b", q, flags=re.IGNORECASE)
        subs = [p.strip(" ?.") for p in parts if len(p.strip()) > 8]
        if not subs:
            subs = [q.strip(" ?.")]
        subs = subs[: self.max_sub_queries]
        return "\n".join(f"- {s}" for s in subs)

    def _summarize(self, content: str) -> str:
        evidence = _parse_evidence(content)
        if not evidence:
            return "NO_EVIDENCE: no supporting chunks were provided."
        # Summarize by quoting the first sentence of the top evidence chunk and
        # attaching the chunk IDs it draws from.
        top = evidence[: min(2, len(evidence))]
        first = top[0].text.split(". ")[0].strip()
        cids = ", ".join(e.chunk_id for e in top)
        return f"{first}. (chunks: {cids})"

    def _respond(self, content: str) -> str:
        q = self._extract(content, "QUESTION:")
        evidence = _parse_evidence(content)
        if not evidence:
            return (
                "Insufficient evidence: the retrieved documents do not contain "
                "information to answer this question."
            )
        q_kw = set(_keywords(q))
        # Compose an answer referencing the most on-topic evidence chunks.
        lines: list[str] = []
        used: list[str] = []
        for ev in evidence:
            ev_kw = set(_keywords(ev.text))
            # Convert chunk_id (doc::section::n) into a [doc:section] citation.
            doc_sec = ev.chunk_id.rsplit("::", 1)[0].replace("::", ":")
            citation = f"[{doc_sec}]"
            first_sentence = ev.text.split(". ")[0].strip()
            lines.append(f"{first_sentence}. {citation}")
            used.append(ev.chunk_id)
            if q_kw and not (q_kw & ev_kw) and len(lines) >= 1:
                # keep going but the first relevant chunk anchors the answer
                pass
            if len(lines) >= min(4, len(evidence)):
                break
        return "Based on the retrieved evidence:\n" + "\n".join(lines)

    # ---- helpers ---------------------------------------------------------
    @staticmethod
    def _extract(content: str, marker: str) -> str:
        """Return the text following ``marker`` up to the next blank line."""
        idx = content.find(marker)
        if idx < 0:
            return content.strip()
        rest = content[idx + len(marker) :].strip()
        # Stop at a double newline (evidence block) if present.
        return rest.split("\n\n", 1)[0].strip()
