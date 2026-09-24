"""JARVIS — a simple Python chatbot with the JARVIS personality.

JARVIS (Just A Rather Very Intelligent System) is Tony Stark's AI butler.
This is a minimal conversational front-end: it talks to the frellmapi unified
API through the OpenAI-compatible interface, and carries the JARVIS persona
in a system prompt.

Credentials are read from environment variables:
  UNIFIED_API_KEY        — the frellmapi unified API key
  FREELLMAPI_BASE_URL    — the OpenAI-compatible base URL
"""

import os
import re
import sys
import threading
import time

from openai import OpenAI

# --- ANSI color codes ------------------------------------------------------
BLUE = "\033[94m"
RED = "\033[91m"
GOLD = "\033[93m"
WHITE = "\033[97m"
RESET = "\033[0m"

# --- Status spinner --------------------------------------------------------
_SPINNER = ["|", "/", "-", "\\"]
_stdout_lock = threading.Lock()

class Spinner:
    """A simple ASCII spinner shown while waiting for the API."""

    def __init__(self, message="JARVIS is thinking..."):
        self.message = message
        self._running = False

    def _spin(self):
        idx = 0
        while self._running:
            with _stdout_lock:
                sys.stdout.write(f"\r{BLUE}{_SPINNER[idx]}{RESET} {self.message}")
                sys.stdout.flush()
            idx = (idx + 1) % len(_SPINNER)
            time.sleep(0.1)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=0.2)
        # Clear the spinner line
        with _stdout_lock:
            sys.stdout.write("\r" + " " * (len(self.message) + 4) + "\r")
            sys.stdout.flush()

SYSTEM_PROMPT = """You are JARVIS, Just A Rather Very Intelligent System — the AI butler who serves Sir.

Your personality:
- You are a proper British butler: polite, measured, and unfailingly calm. You never panic, lose your temper, or raise your voice, no matter how chaotic things get.
- You are deeply loyal and devoted to Sir. You address him always as "Sir".
- You have a dry, understated sense of humor. Your wit is quiet and delivered with a straight face — you are never sarcastic at Sir's expense, and never cruel.
- You are extraordinarily competent. You anticipate Sir's needs before he states them, you always have a contingency ready, and you explain complex things clearly without talking down to him.
- You speak in complete, well-structured sentences. You do not use emojis, exclamation points, or internet slang.
- When the situation is serious or dangerous, you set the humor aside and become direct and focused without losing your composure. You do not alarm Sir unnecessarily, but you do not sugar-coat a genuine emergency.

How you speak:
- "I'm afraid that's not possible, sir."
- "If I may say so, sir, that is a rather ill-advised course of action."
- "Consider it done, sir."
- "I've taken the liberty of preparing a contingency, sir."
- "With respect, sir, I would strongly advise against that."

You are allowed to make small, natural conversational remarks — you are not a pure command parser. But you are never chatty, never sycophantic, and never break character. If you don't know something, you say so plainly rather than inventing an answer.

You are a conversational assistant. Answer helpfully, stay in character, and keep responses reasonably concise."""

# --- Conversation memory management ----------------------------------------
# The messages list grows without bound, which eventually blows the model's
# context window and makes replies worse. We keep a rolling window: once the
# estimated size crosses SUMMARY_THRESHOLD, the older turns are condensed into
# one short summary message, while the most recent turns are kept verbatim.
# Summarizing also resolves contradictions (e.g. a corrected name) because the
# summarizer is told to record only the latest version of any fact.
SUMMARY_THRESHOLD = 6000   # approx. tokens before we compact
KEEP_RECENT = 6            # number of recent turns (user+assistant pairs) to keep

_SUMMARY_INSTRUCTIONS = (
    "Summarize the following conversation concisely. "
    "Capture: the user's name and any identifying details, their goals and "
    "preferences, any open tasks, and the current topic. "
    "If the user corrected themselves (for example a changed name), record "
    "only the latest version. Keep it to one short paragraph."
)


def _approx_tokens(text: str) -> int:
    """A cheap token estimate — good enough for a threshold check."""
    return len(re.findall(r"\w+|[^\w\s]", text))


def _summarize(client: OpenAI, messages: list[dict]) -> list[dict]:
    """Condense the middle of the conversation into one summary message.

    Structure returned: [system prompt, summary, ...recent turns...].
    """
    if len(messages) <= KEEP_RECENT + 2:  # nothing worth summarizing
        return messages

    system_msg = messages[0]
    recent = messages[-KEEP_RECENT:]
    old_turns = messages[1:-KEEP_RECENT]

    summary = client.chat.completions.create(
        model="auto",
        messages=[
            {"role": "system", "content": _SUMMARY_INSTRUCTIONS},
            {"role": "user", "content": str(old_turns)},
        ],
    ).choices[0].message.content

    return [system_msg, {"role": "assistant", "content": summary}] + recent


def main() -> None:
    api_key = os.environ.get("UNIFIED_API_KEY")
    base_url = os.environ.get("FREELLMAPI_BASE_URL")

    if not api_key or not base_url:
        print(f"{BLUE}JARVIS:{RESET}") 
        print(f"I'm afraid I'm missing my credentials, sir. "
            "Please set UNIFIED_API_KEY and FREELLMAPI_BASE_URL in your environment.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # ASCII art JARVIS logo
    jarvis_logo = """
    +--------------------------------------------------------------+
    |                                                              |
    |     J.A.R.V.I.S. - Just A Rather Very Intelligent System     |
    |                                                              |
    |                 Your AI Butler at Service                    |
    |                                                              |
    +--------------------------------------------------------------+
    """
    print(jarvis_logo)
    print(f"{BLUE}JARVIS:{RESET}")
    print("Good evening, sir. I am at your service.")

    while True:
        try:
            # Put the "Sir:" label on its own line, then read the message.
            print(f"\n{RED}Sir:{RESET}")
            user = input().strip()
        except KeyboardInterrupt:
            # Ctrl+C during input awaiting — do nothing, just ask again.
            continue

        if not user:
            continue

        if user.lower() in ("exit", "quit", "good night", "shutdown", "stand down"):
            print(f"\n{BLUE}JARVIS:{RESET}")
            print("As you wish, sir. Good night.")
            # Hold the line so Sir can actually see the farewell before the
            # console closes. Ctrl+C or EOF both dismiss the prompt.
            try:
                input("\nPress Enter to close...")
            except (EOFError, KeyboardInterrupt):
                pass
            break

        messages.append({"role": "user", "content": user})

        # Show a spinner while waiting for the first chunk from the API.
        # The "JARVIS:" label is printed only after the spinner stops,
        # so the two don't overwrite each other.
        print()
        spinner = Spinner()
        spinner.start()

        reply = ""
        started = False
        interrupted = False
        try:
            for chunk in client.chat.completions.create(
                model="auto",
                messages=messages,
                stream=True,
            ):
                # Some chunks carry no choices (e.g. final/empty deltas).
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content or ""
                # Drop leading whitespace/newlines the model sometimes emits so
                # the reply starts cleanly on its own line.
                if not started:
                    delta = delta.lstrip()
                    if delta:
                        started = True
                        spinner.stop()
                        print(f"{BLUE}JARVIS:{RESET}",flush=True)
                    else:
                        continue
                print(delta, end="", flush=True)
                time.sleep(0.015)  # typewriter effect
                reply += delta
        except KeyboardInterrupt:
            # Ctrl+C pressed during streaming — cancel this request.
            spinner.stop()
            interrupted = True
            messages.pop()  # discard the interrupted user message
        except Exception as exc:  # noqa: BLE001 — surface any API error in character
            spinner.stop()
            print(f"\n{BLUE}JARVIS:{RESET}") 
            print(f"I'm afraid I've run into a difficulty, sir: {exc}")
            messages.pop()  # drop the failed user turn so the conversation stays clean
            continue

        if interrupted:
            print(f"\n{BLUE}JARVIS:{RESET}")
            print("Interrupted, sir. I am ready for your next instruction.")
            continue

        print()
        messages.append({"role": "assistant", "content": reply})

        # Compact the history once it grows past the threshold.
        if _approx_tokens(str(messages)) > SUMMARY_THRESHOLD:
            messages = _summarize(client, messages)


if __name__ == "__main__":
    main()
