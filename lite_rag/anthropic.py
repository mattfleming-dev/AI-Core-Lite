"""Anthropic Messages API <-> OpenAI Chat Completions translation (images, tools, streaming)."""
import json
import uuid

STOP_REASONS = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use", "function_call": "tool_use",
                "content_filter": "end_turn"}


def _text_of(blocks):
    if isinstance(blocks, str):
        return blocks
    return "\n".join(b.get("text", "") for b in blocks or [] if isinstance(b, dict) and b.get("type") == "text")


def _image_part(block):
    source = block.get("source", {})
    if source.get("type") == "base64":
        url = f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
    else:
        url = source.get("url", "")
    return {"type": "image_url", "image_url": {"url": url}}


def to_openai_request(body):
    messages = []
    system = _text_of(body.get("system"))
    if system:
        messages.append({"role": "system", "content": system})
    for message in body.get("messages", []):
        role, content = message.get("role"), message.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        parts, tool_calls, tool_results = [], [], []
        for block in content or []:
            kind = block.get("type")
            if kind == "text":
                parts.append({"type": "text", "text": block.get("text", "")})
            elif kind == "image":
                parts.append(_image_part(block))
            elif kind == "tool_use":
                tool_calls.append({"id": block.get("id"), "type": "function",
                                   "function": {"name": block.get("name"), "arguments": json.dumps(block.get("input", {}))}})
            elif kind == "tool_result":
                inner = block.get("content")
                text = inner if isinstance(inner, str) else _text_of(inner)
                tool_results.append({"role": "tool", "tool_call_id": block.get("tool_use_id"), "content": text})
        messages.extend(tool_results)
        if role == "assistant" and tool_calls:
            text = "\n".join(p["text"] for p in parts if p["type"] == "text")
            messages.append({"role": "assistant", "content": text or None, "tool_calls": tool_calls})
        elif parts:
            only_text = all(p["type"] == "text" for p in parts)
            messages.append({"role": role, "content": "\n".join(p["text"] for p in parts) if only_text else parts})
    out = {"model": body.get("model", ""), "messages": messages, "stream": bool(body.get("stream"))}
    for source, target in (("max_tokens", "max_tokens"), ("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k")):
        if body.get(source) is not None:
            out[target] = body[source]
    if body.get("stop_sequences"):
        out["stop"] = body["stop_sequences"]
    if body.get("tools"):
        out["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                                           "parameters": t.get("input_schema", {"type": "object", "properties": {}})}}
                        for t in body["tools"]]
        choice = body.get("tool_choice") or {}
        if choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "tool":
            out["tool_choice"] = {"type": "function", "function": {"name": choice.get("name")}}
        elif choice.get("type") == "none":
            out["tool_choice"] = "none"
    if out["stream"]:
        out["stream_options"] = {"include_usage": True}
    return out


def to_anthropic_response(data, model):
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message", {})
    content = []
    if message.get("content"):
        content.append({"type": "text", "text": message["content"]})
    for call in message.get("tool_calls") or []:
        try:
            arguments = json.loads(call["function"].get("arguments") or "{}")
        except ValueError:
            arguments = {}
        content.append({"type": "tool_use", "id": call.get("id") or f"toolu_{uuid.uuid4().hex[:20]}",
                        "name": call["function"]["name"], "input": arguments})
    usage = data.get("usage", {})
    return {"id": data.get("id") or f"msg_{uuid.uuid4().hex[:20]}", "type": "message", "role": "assistant",
            "model": data.get("model") or model, "content": content or [{"type": "text", "text": ""}],
            "stop_reason": STOP_REASONS.get(choice.get("finish_reason"), "end_turn"), "stop_sequence": None,
            "usage": {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)}}


def stream_to_anthropic(chunks, model):
    """Yield (event_name, payload) from OpenAI stream chunk dicts."""
    message_id = f"msg_{uuid.uuid4().hex[:20]}"
    yield "message_start", {"type": "message_start", "message": {
        "id": message_id, "type": "message", "role": "assistant", "model": model, "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0}}}
    index, open_kind, tool_slots, finish, usage = -1, None, {}, None, {}

    def close():
        nonlocal open_kind
        if open_kind is not None:
            open_kind = None
            return [("content_block_stop", {"type": "content_block_stop", "index": index})]
        return []

    for chunk in chunks:
        usage = chunk.get("usage") or usage
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                if open_kind != "text":
                    yield from close()
                    index += 1
                    open_kind = "text"
                    yield "content_block_start", {"type": "content_block_start", "index": index,
                                                  "content_block": {"type": "text", "text": ""}}
                yield "content_block_delta", {"type": "content_block_delta", "index": index,
                                              "delta": {"type": "text_delta", "text": delta["content"]}}
            for call in delta.get("tool_calls") or []:
                slot = call.get("index", 0)
                if slot not in tool_slots:
                    yield from close()
                    index += 1
                    open_kind = "tool"
                    tool_slots[slot] = index
                    yield "content_block_start", {"type": "content_block_start", "index": index, "content_block": {
                        "type": "tool_use", "id": call.get("id") or f"toolu_{uuid.uuid4().hex[:20]}",
                        "name": (call.get("function") or {}).get("name", ""), "input": {}}}
                arguments = (call.get("function") or {}).get("arguments")
                if arguments:
                    yield "content_block_delta", {"type": "content_block_delta", "index": tool_slots[slot],
                                                  "delta": {"type": "input_json_delta", "partial_json": arguments}}
            finish = choice.get("finish_reason") or finish
    yield from close()
    yield "message_delta", {"type": "message_delta", "delta": {"stop_reason": STOP_REASONS.get(finish, "end_turn"), "stop_sequence": None},
                            "usage": {"output_tokens": usage.get("completion_tokens", 0)}}
    yield "message_stop", {"type": "message_stop"}


def models_to_anthropic(data):
    return {"data": [{"type": "model", "id": m.get("id"), "display_name": m.get("id"), "created_at": "2024-01-01T00:00:00Z"}
                     for m in data.get("data", [])], "has_more": False, "first_id": None, "last_id": None}


def error_body(message, kind="api_error"):
    return {"type": "error", "error": {"type": kind, "message": message}}
