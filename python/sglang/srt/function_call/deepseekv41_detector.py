from typing import Any, Dict, List, Optional

from sglang.srt.entrypoints.openai.encoding_dsv41 import (
    _split_tool_name,
    _tool_name_for_encoding,
)
from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.core_types import StreamingParseResult, ToolCallItem
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector


def _qualified_tool_name(tool: Tool) -> Optional[str]:
    """The name the V4.1 prompt shows for `tool`: `namespace::name` when namespaced.

    sglang-dsv41-encoding-namespace-turns.py: the tool-level namespace wins over
    the function's, as in the reference encoder.
    """
    function = tool.function
    namespace = getattr(tool, "namespace", None)
    if namespace is None:
        namespace = getattr(function, "namespace", None)
    try:
        return _tool_name_for_encoding({"name": function.name, "namespace": namespace})
    except (ValueError, TypeError):
        return function.name


def _split_namespaces(calls: List[ToolCallItem]) -> List[ToolCallItem]:
    """`ns::name` -> name="name", namespace="ns" on each call that carries a name."""
    for call in calls:
        if call.name and "::" in call.name:
            try:
                call.namespace, call.name = _split_tool_name(call.name)
            except ValueError:
                pass
    return calls


class DeepSeekV41Detector(DeepSeekV32Detector):
    """DeepSeek V4.1 DSML detector.

    Tag names have a leading space: " calls", " invoke", and " parameter".
    """

    tool_calls_block_name = " calls"
    invoke_tag_name = " invoke"
    parameter_tag_name = " parameter"

    def _get_tool_indices(self, tools: List[Tool]) -> Dict[str, int]:
        # The model calls a namespaced tool by its qualified name.
        return {
            name: i for i, tool in enumerate(tools) if (name := _qualified_tool_name(tool))
        }

    def parse_base_json(self, action: Any, tools: List[Tool]) -> List[ToolCallItem]:
        return _split_namespaces(super().parse_base_json(action, tools))

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        result = super().parse_streaming_increment(new_text, tools)
        _split_namespaces(result.calls)
        return result

    def get_structural_tag_name(self) -> Optional[str]:
        # xgrammar's builtin "deepseek_v4" tag is the unspaced grammar; fall
        # back to structure_info so constrained decoding uses the spaced tags.
        return None
