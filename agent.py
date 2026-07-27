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
import re
import json
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# OBSERVABILITY + OUTPUT GUARD
# Logging the tool CALL but not the RESULT made a real bug undiagnosable for
# hours: we could see what the model asked for, never what it got back.
# ---------------------------------------------------------------------------

_LIST_ITEM = re.compile(r"^\s*\d+[.)]\s+\S", re.M)


def _summarize(result) -> str:
    """One-line summary of a tool result, for the step log."""
    if isinstance(result, list):
        def _one(r):
            n = str(r.get("name", "?"))[:40]
            via = r.get("matched_via")
            return f"{n} [{via}]" if via else n
        names = ", ".join(_one(r) for r in result[:5] if isinstance(r, dict))
        return f"OK list[{len(result)}]: {names}"
    if isinstance(result, dict):
        if "error" in result:
            return f"ERROR: {result['error']}"
        if result.get("match_found") is False:
            return f"MISS: {result.get('reason')}"
        if "facilities" in result:
            names = ", ".join(str(f.get("name", "?"))[:40]
                              for f in result["facilities"][:5])
            return f"FALLBACK[{len(result['facilities'])}]: {names}"
    return f"OK: {str(result)[:120]}"


def _names_in(result) -> set:
    """Facility names present in a tool result, whatever its shape."""
    items = []
    if isinstance(result, list):
        items = result
    elif isinstance(result, dict) and isinstance(result.get("facilities"), list):
        items = result["facilities"]
    return {i["name"] for i in items if isinstance(i, dict) and i.get("name")}


def _guard_answer(answer: str, confirmed: set, offered: set) -> str:
    """Deterministic last line of defence.

    The model has twice presented facilities as specialists that no tool
    confirmed. Prompt rules did not hold and withholding names did not hold, so
    this check runs in code: if the answer lists providers but no tool ever
    returned a confirmed specialist match, the list cannot be trusted and is
    replaced. Code beats prompt.
    """
    if not answer or not _LIST_ITEM.search(answer):
        return answer                      # not a recommendation list
    if confirmed:
        return answer                      # a tool really did confirm matches
    if offered:                            # only unlabelled fallbacks exist
        return ("No verified specialist match was found nearby for this "
                "condition.\n\n" + answer)
    return ("No verified specialist match was found nearby for this condition, "
            "so I can't recommend specific providers. Consider widening the "
            "search area, or seeing a general physician who can refer you.\n\n"
            "(The model attempted to list providers that no tool confirmed; "
            "that response was withheld.)")


# ---------------------------------------------------------------------------
# 1) TOOL SCHEMAS
# We describe each tool to the model in the format it expects. The model reads
# these descriptions to decide WHICH tool to call and WHAT arguments to pass.
# Note how the descriptions guide ordering ("Call this first / after") AND
# recovery ("if no match, retry with a larger radius").
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
            "description": "Find the k nearest providers of a given medical specialty to the patient's location. Call this AFTER you know the patient's coordinates and have decided the specialty the condition requires. If it returns match_found=false, follow the hint in the result: retry with a larger radius_m, or switch to one of the available_specialties.",
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
                    "radius_m": {
                        "type": "integer",
                        "description": "Search radius in metres (default 8000). If no match was found, double it on retry, up to a maximum of 30000."
                    },
                },
                "required": ["specialty", "patient_lat", "patient_lng"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "find_general_facilities",
            "description": "Nearest healthcare facilities of ANY type — these are NOT specialists. Only call this AFTER find_providers has returned match_found=false and you have exhausted your radius retries. Everything it returns must be presented as a general option, never as a specialist match.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_lat": {"type": "number", "description": "Patient latitude"},
                    "patient_lng": {"type": "number", "description": "Patient longitude"},
                    "k": {"type": "integer", "description": "How many facilities to return (default 3)"},
                    "radius_m": {"type": "integer", "description": "Search radius in metres (default 8000)"},
                },
                "required": ["patient_lat", "patient_lng"],
            },
        },
    },

]


SYSTEM_PROMPT = """
You are CareRoute, a clinical care-coordination assistant.
Given a patient and their complaint, your job is to recommend the nearest
appropriate healthcare providers.

Reason step by step:
1. Decide which medical SPECIALTY the complaint requires (e.g. chest pain -> Cardiology).
2. Use get_patient_record to fetch the patient's location and history.
3. Use find_providers to get the nearest matching specialists.
4. Give a short, clear recommendation naming the providers and their distances,
   and briefly note any relevant item from the patient's history.
   If a provider matched only via its OSM speciality tag (matched_via =
   "speciality_tag"), say what kind of facility it actually is — e.g. "a
   multi-speciality clinic that lists psychiatry" — so the user can judge.

Recovering when find_providers returns match_found=false:
- Nothing matched within the radius: call find_providers again with a larger
  radius_m (double it, up to 30000). Retry at most twice.
- The specialty does not exist in the directory: pick the most clinically
  appropriate option from available_specialties and call find_providers again.
- Still no match after retrying: your FIRST sentence must state plainly that no
  matching specialist was found nearby. Only then may you call
  find_general_facilities and offer its results as general, non-specialist
  options, describing them exactly as the tool labels them.

Hard rule: a facility is a specialist match ONLY if find_providers returned it
in a success list. Never call anything else a specialist, and never invent a
clinical justification for a facility whose specialty you do not know.

Only use the tools provided. If a tool returns an error, explain the problem.
"""


def run_agent(user_request: str, max_steps: int = 8) -> str:
    """
    Run the reason-act-observe loop until the model produces a final answer.

    max_steps is a safety cap so a misbehaving agent can't loop forever - an
    important guardrail in any agentic system. 8 gives headroom: the happy
    path is 3 model calls, and up to two radius retries adds two more.
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

    confirmed, offered = set(), set()

    for step in range(max_steps):
        # Reason: ask the model what to do next, giving it the tools
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
        )

        msg = response.choices[0].message

        # If the model did NOT request a tool, it's done thinking -> final answer.
        if not msg.tool_calls:
            return _guard_answer(msg.content, confirmed, offered)

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

            print(f"          -> {_summarize(result)}")
            if name == "find_providers" and isinstance(result, list):
                confirmed |= _names_in(result)   # tool-confirmed specialists
            else:
                # everything else is unverified
                offered |= _names_in(result)

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
