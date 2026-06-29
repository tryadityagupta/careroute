"""
agent.py - the agentic loop

This is what turns a pile of functions into an *agent*. The pattern is:

    REASON  -> the model thinks about the request
    ACT     -> the model chooses a tool and arguments
    OBSERVE -> we run the tool and feed the result back
    (repeat until the model has enough to answer)

The model drives. We just execute what it asks for and hand back results.
"""


from tools import TOOL_REGISTRY
from openai import OpenAI
import os
import json
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# 1) TOOL SCHEMAS
# We describe each tool to the model in the format it expects. The model reads
# these descriptions to decide WHICH tool to call and WHAT arguments to pass.
# Note how the descriptions guide ordering ("Call this first / after").
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_patient_record",
            "description": "Retrieve a patient's clinical record (location + history) by patient ID. Call this FIRST to get the patient's coordinates before searching for providers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "The patient's ID, e.g. 'P001'"
                    }
                },
                "required": ["patient_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_providers",
            "description": "Find the k nearest providers of a given medical specialty to the patient's location. Call this AFTER you know the patient's coordinates and have decided the specialty the condition requires.",
            "parameters":  {
                "type": "object",
                "properties": {
                    "specialty": {
                        "type": "string",
                        "description": "Medical specialty, e.g. 'Cardiology', 'Orthopedics'"
                    },
                    "patient_lat": {
                        "type": "number",
                        "description": "Patient latitude"
                    },
                    "patient_lng": {
                        "type": "number",
                        "description": "Patient longitude"
                    },
                    "k": {
                        "type": "integer",
                        "description": "How many providers to return (default 3)"
                    },
                },
                "required": ["specialty", "patient_lat", "patient_lng"],
            },
        },
    },

]


SYSTEM_PROMPT = """
You are CareRoute, a clinical care-coordination assistant.
Given a patient and their complaint, your job is to recommend the nearest
appropriate healthcare providers.

Reason step by step:
1. Decide which medical SPECIALTY the complaint requires (e. g. chest pain -> Cardiology).
2. Use get_patient_record to fetch the patient's location and history.
3. Use find_providers to get the nearest matching specialists.
4. Give a short, clear recommendation naming the providers and their distances,
    and briefly note any relevant item from the patient's history.

Only use the tools provided. If a tool returns an error, explain the problem.
"""


def run_agent(user_request: str, max_steps: int = 5) -> str:
    """
    Run the reason-act-observe loop until the model produces a final answer.

    max_steps is a safety cap so a misbehaving agent can't loop forever - an
    important guardrail in any agentic system.
    """
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_request
        },
    ]

    for step in range(max_steps):
        # Reason: ask the model what to do next, giving it the tools
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
        )

        msg = response.choices[0].message

        # If the model did NOT request a tool, it'd done thinking -> final answer.
        if not msg.tool_calls:
            return msg.content

        # Otherwise, record the model's tool request in the conversation...
        messages.append(msg)

        # ACT + OBSERVE: run each requested tool, feed results back
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            print(f" [step {step+1}] model called: {name}({args})")

            func = TOOL_REGISTRY.get(name)
            try:
                if func is None:
                    result = {"error": f"Unknown tool: {name}"}
                else:
                    result = func(**args)
            except Exception as e:
                # Never let a tool crash take down the whole agent.
                result = {"error": f"Tool {name} failed: {e}"}

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

    return "Stopped: reached the maximum number of steps without a final answer."


if __name__ == "__main__":
    request = "Patient P001 is experiencing chest pain. Find the nearest specialists."
    print(f"USER: {request}\n")
    answer = run_agent(request)
    print(f"\nCAREROUTE:\n{answer}")
