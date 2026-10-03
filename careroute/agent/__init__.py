"""The LangGraph agent.

    graph.py         CareRouteAgent — the reason/act/observe graph, single-shot + multi-turn
    prompts.py       the system prompt and the emergency context
    state.py         CareRouteState — what every node reads and writes
    toolkit.py       LangChain adapters: the tool names/descriptions the model sees
    guard.py         AnswerGuard — deterministic last check on the final answer
    messages.py      helpers to read tool results and message text
    tracing.py       TurnTracer — privacy-trimmed record of what a turn did
    checkpointer.py  CheckpointerFactory — MemorySaver or PostgresSaver
"""
