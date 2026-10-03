"""CLI demo:  python -m careroute "Patient P001 has chest pain. Find specialists."

Uses the dummy JSON backend unless USE_REAL_PROVIDERS is set, and the real
model (OPENAI_API_KEY / OPENAI_BASE_URL)."""

import sys

from careroute.container import Container


def main() -> None:
    request = " ".join(sys.argv[1:]) or (
        "Patient P001 is experiencing chest pain. Find the nearest specialists.")
    print(f"USER: {request}\n")
    answer, _ = Container().agent.run_single(request)
    print(f"\nCAREROUTE:\n{answer}")


if __name__ == "__main__":
    main()
