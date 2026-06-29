"""
Verify the agent LOOP without calling OpenAI, by faking the model's replies.
"""

import json
import types
import agent

# Build fake response objects shaped like the OpenAI SDK's return value.


def fake_tool_call(call_id, name, args):
    fn = types.SimpleNamespace(name=name, arguments=json.dumps(args))
    return types.SimpleNamespace(id=call_id, function=fn)


def fake_message(content=None, tool_calls=None):
    return types.SimpleNamespace(content=content, tool_calls=tool_calls)


def fake_response(message):
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


# Scripted "model": step 1 -> get record, step 2 -> find providers, step 3 -> answer.
scripted = [
    fake_response(fake_message(tool_calls=[
        fake_tool_call("c1", "get_patient_record", {"patient_id": "P001"})
    ])),
    fake_response(fake_message(tool_calls=[
        fake_tool_call("c2", "find_providers", {
            "specialty": "Cardiology", "patient_lat": 12.9352, "patient_lng": 77.6245, "k": 3
        })
    ])),
    fake_response(fake_message(content=(
        "For Ramesh Kumar (history: Hypertension), the nearest cardiologists are: "
        "Dr. Meera Iyer (HSR Cardiology Center, 3.05 km), "
        "Dr. Imran Khan (Jayanagar Heart Institute, 3.56 km), "
        "Dr. Anjali Rao (Indiranagar Heart Clinic, 4.46 km). "
        "Given his hypertension, prompt cardiac evaluation is advised."
    ))),
]

_calls = {"n": 0}


def fake_create(*a, **kw):
    r = scripted[_calls["n"]]
    _calls["n"] += 1
    return r


# Swap the real OpenAI call for our scripted one, then run the real loop.
agent.client.chat.completions.create = fake_create

print("USER: Patient P001 is experiencing chest pain. Find the nearest specialists.\n")
print("CAREROUTE:\n" + agent.run_agent(
    "Patient P001 is experiencing chest pain. Find the nearest specialists."))
