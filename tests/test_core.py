from llm_toolkit import ask
from llm_toolkit.adapters import Message, Response, Usage


def test_imports_smoke() -> None:
    assert callable(ask)
    message = Message(role="user", content="hello")
    usage = Usage(input_tokens=1, output_tokens=2)
    response = Response(provider="gpt", model="gpt-4o-mini", text="hi", usage=usage)
    assert message.role == "user"
    assert response.usage.output_tokens == 2
