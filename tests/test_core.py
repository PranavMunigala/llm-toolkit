from llm_toolkit import ask
from llm_toolkit.adapters import Message, Response, Usage


def test_imports_smoke() -> None:
    assert callable(ask)
    message = Message(role="user", content="hello")
    usage = Usage(input_tokens=1, output_tokens=2)
    response = Response(provider="gpt", model="gpt-4o-mini", text="hi", usage=usage)
    assert message.role == "user"
    assert response.usage.output_tokens == 2


#python -m llm_toolkit.cli "What is DNA to RNA transcription. Please explain in simple terms and only two sentences maximum" --provider gemini --model "gemini-3.5-flash" --stream
#python -m llm_toolkit.cli "How can you find the Origin of replication in a genome. Please explain under two sentences." --provider gpt --model "gpt-4o"

#python -m llm_toolkit.cli "Calculate 365 multiplied by 263" --provider gpt --model "gpt-4o"