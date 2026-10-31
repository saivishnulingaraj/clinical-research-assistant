"""Amazon Bedrock adapter (Nova Pro) via langchain-aws.

The model ID and AWS region are injected from :class:`Settings` — never
hard-coded. This module is imported lazily (only when ``CRA_LLM_PROVIDER=bedrock``)
so the sandbox/CI, which cannot reach AWS, never needs the dependency at import
time.
"""

from __future__ import annotations

from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential

from cra.llm.base import BaseLLM, LLMResponse, Message, estimate_tokens


@dataclass
class BedrockLLM(BaseLLM):
    """Calls Amazon Nova Pro on Bedrock through ``langchain_aws.ChatBedrock``."""

    model_id: str = ""
    region: str = ""

    def __post_init__(self) -> None:
        if not self.model_id or not self.region:
            raise ValueError("BedrockLLM requires a model_id and region (from Settings)")
        # Imported here so langchain-aws/boto3 are only needed for real runs.
        from langchain_aws import ChatBedrock

        self._client = ChatBedrock(
            model_id=self.model_id,
            region_name=self.region,
            model_kwargs={
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    )
    def _generate(self, messages: list[Message]) -> LLMResponse:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        role_map = {
            "system": SystemMessage,
            "user": HumanMessage,
            "assistant": AIMessage,
        }
        lc_messages = [role_map.get(m.role, HumanMessage)(content=m.content) for m in messages]

        result = self._client.invoke(lc_messages)
        text = result.content if isinstance(result.content, str) else str(result.content)

        # Prefer real usage metadata from Bedrock; fall back to an estimate.
        usage = getattr(result, "usage_metadata", None) or {}
        input_tokens = int(usage.get("input_tokens", 0)) or estimate_tokens(
            "\n".join(m.content for m in messages)
        )
        output_tokens = int(usage.get("output_tokens", 0)) or estimate_tokens(text)

        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
