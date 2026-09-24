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
import sys

from openai import OpenAI

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


def main() -> None:
    api_key = os.environ.get("UNIFIED_API_KEY")
    base_url = os.environ.get("FREELLMAPI_BASE_URL")

    if not api_key or not base_url:
        print(
            "JARVIS: I'm afraid I'm missing my credentials, sir. "
            "Please set UNIFIED_API_KEY and FREELLMAPI_BASE_URL in your environment.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("\nJARVIS: Good evening, sir. I am at your service.")

    while True:
        try:
            # Put the "Sir:" label on its own line, then read the message.
            print("\nSir:")
            user = input().strip()
        except (EOFError, KeyboardInterrupt):
            print("\nJARVIS: Very good, sir. I shall standby.")
            break

        if not user:
            continue

        if user.lower() in ("exit", "quit", "good night", "shutdown", "stand down"):
            print("\nJARVIS: As you wish, sir. Good night.")
            break

        messages.append({"role": "user", "content": user})

        print("\nJARVIS:")
        reply = ""
        started = False
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
                    else:
                        continue
                print(delta, end="", flush=True)
                reply += delta
        except Exception as exc:  # noqa: BLE001 — surface any API error in character
            print(f"\nJARVIS: I'm afraid I've run into a difficulty, sir: {exc}")
            messages.pop()  # drop the failed user turn so the conversation stays clean
            continue

        print()
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
