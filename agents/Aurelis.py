"""
Aurelis - Voice Assistant with Tools (Push-to-Talk)
------------------------------------------------------------
Hold SPACE to record your question, release to get a spoken answer.
Now with tool use: open apps, open websites, check time, list files.
"""

import os

# --- Fix: point directly at ffmpeg since Windows PATH isn't picking it up ---
FFMPEG_BIN = r"C:\Users\luise\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0-full_build\bin"
os.environ["PATH"] = FFMPEG_BIN + os.pathsep + os.environ["PATH"]
# -----------------------------------------------------------------------

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import whisper
import pyttsx3
import keyboard
import anthropic
import tempfile
import subprocess
import webbrowser
import datetime
import json
from dotenv import load_dotenv
from duckduckgo_search import DDGS

load_dotenv()

SAMPLE_RATE = 16000
CONVO_HISTORY = []
BASE_SYSTEM_PROMPT = "You are Aurelis, a concise and helpful voice assistant. Keep responses short and conversational since they'll be spoken aloud. Use tools when the user asks you to do something on their computer, rather than just describing how they'd do it themselves."

# ---------------------------------------------------------------------------
# MEMORY: a small, capped list of durable facts (name, preferences, etc.)
# Kept deliberately separate from CONVO_HISTORY, which stays in-memory only
# so per-session API cost/context doesn't grow across restarts.
# ---------------------------------------------------------------------------

MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aurelis_memory.json")
MAX_FACTS = 30


def load_facts():
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("facts", [])
    except Exception:
        return []


def save_facts(facts):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump({"facts": facts}, f, indent=2)


def add_fact(fact):
    facts = load_facts()
    fact = fact.strip()
    if not fact or fact in facts:
        return facts
    facts.append(fact)
    facts = facts[-MAX_FACTS:]  # cap so the system prompt can't grow unbounded
    save_facts(facts)
    return facts


def build_system_prompt():
    facts = load_facts()
    if not facts:
        return BASE_SYSTEM_PROMPT
    facts_block = "\n".join(f"- {f}" for f in facts)
    return f"{BASE_SYSTEM_PROMPT}\n\nThings you remember about the user:\n{facts_block}"

# ---------------------------------------------------------------------------
# TOOLS: each tool needs (1) a schema Claude sees, (2) a matching Python function
# ---------------------------------------------------------------------------

KNOWN_APPS = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "chrome": "chrome.exe",
    "explorer": "explorer.exe",
    "paint": "mspaint.exe",
    "word": "winword.exe",
    "excel": "excel.exe",
    "spotify": "spotify.exe",
}

TOOLS = [
    {
        "name": "open_app",
        "description": "Open a desktop application on the user's Windows PC by name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "app_name": {
                    "type": "string",
                    "description": "Name of the app, e.g. 'notepad', 'chrome', 'calculator', 'spotify'",
                }
            },
            "required": ["app_name"],
        },
    },
    {
        "name": "open_website",
        "description": "Open a website in the default browser.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL to open, e.g. 'https://youtube.com'",
                }
            },
            "required": ["url"],
        },
    },
    {
        "name": "get_current_time",
        "description": "Get the current date and time.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_files",
        "description": "List files in a folder on the user's PC.",
        "input_schema": {
            "type": "object",
            "properties": {
                "folder_path": {
                    "type": "string",
                    "description": "Full path to the folder, e.g. 'C:\\\\Users\\\\luise\\\\Downloads'",
                }
            },
            "required": ["folder_path"],
        },
    },
        {
        "name": "web_search",
        "description": "Search the web for current information, news, facts, or anything the assistant doesn't already know.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "remember_fact",
        "description": "Save a short durable fact about the user for future conversations (e.g. their name, a preference, an important detail). Only use this when the user asks you to remember something or shares info clearly worth keeping long-term — not for one-off details.",
        "input_schema": {
            "type": "object",
            "properties": {
                "fact": {
                    "type": "string",
                    "description": "The fact to remember, written concisely, e.g. 'User's name is Luis' or 'Prefers metric units'",
                }
            },
            "required": ["fact"],
        },
    },
]


def tool_open_app(app_name):
    key = app_name.strip().lower()
    exe = KNOWN_APPS.get(key)
    if not exe:
        return f"I don't know how to open '{app_name}'. Known apps: {', '.join(KNOWN_APPS)}"
    try:
        subprocess.Popen(exe, shell=True)
        return f"Opened {app_name}."
    except Exception as e:
        return f"Failed to open {app_name}: {e}"


def tool_open_website(url):
    try:
        webbrowser.open(url)
        return f"Opened {url} in your browser."
    except Exception as e:
        return f"Failed to open {url}: {e}"


def tool_get_current_time():
    now = datetime.datetime.now()
    return now.strftime("%A, %B %d, %Y at %I:%M %p")


def tool_list_files(folder_path):
    try:
        entries = os.listdir(folder_path)
        if not entries:
            return f"{folder_path} is empty."
        return f"Found {len(entries)} items: " + ", ".join(entries[:25])
    except Exception as e:
        return f"Couldn't list files at {folder_path}: {e}"


def tool_web_search(query):
    try:
        results = DDGS().text(query, max_results=5)
        if not results:
            return f"No results found for '{query}'."
        summary = []
        for r in results:
            summary.append(f"{r['title']}: {r['body']}")
        return " | ".join(summary)
    except Exception as e:
        return f"Web search failed: {e}"


def tool_remember_fact(fact):
    if not fact.strip():
        return "No fact given."
    add_fact(fact)
    return f"Got it, I'll remember: {fact}"


def execute_tool(name, tool_input):
    if name == "open_app":
        return tool_open_app(tool_input.get("app_name", ""))
    if name == "open_website":
        return tool_open_website(tool_input.get("url", ""))
    if name == "web_search":
        return tool_web_search(tool_input.get("query", ""))
    if name == "get_current_time":
        return tool_get_current_time()
    if name == "list_files":
        return tool_list_files(tool_input.get("folder_path", ""))
    if name == "remember_fact":
        return tool_remember_fact(tool_input.get("fact", ""))
    return f"Unknown tool: {name}"


print("Loading Whisper model (this happens once)...")
whisper_model = whisper.load_model("base")

tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 180)

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise RuntimeError(
        "ANTHROPIC_API_KEY not found. Create a .env file next to this script "
        "with a line: ANTHROPIC_API_KEY=sk-ant-..."
    )
client = anthropic.Anthropic(api_key=api_key)


def record_audio():
    print("\n🎙️  Listening... (hold SPACE)")
    frames = []

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=callback)
    stream.start()
    while keyboard.is_pressed("space"):
        sd.sleep(50)
    stream.stop()
    stream.close()

    if not frames:
        return None
    audio = np.concatenate(frames, axis=0)
    return audio


def transcribe(audio):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav.write(f.name, SAMPLE_RATE, audio)
        result = whisper_model.transcribe(f.name, fp16=False)
    return result["text"].strip()


def ask_claude(user_text):
    CONVO_HISTORY.append({"role": "user", "content": user_text})

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            system=build_system_prompt(),
            tools=TOOLS,
            messages=CONVO_HISTORY,
        )

        # Save Claude's full response (may include text + tool_use blocks)
        CONVO_HISTORY.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  [running tool: {block.name}({block.input})]")
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            # Send tool results back so Claude can respond with them
            CONVO_HISTORY.append({"role": "user", "content": tool_results})
            continue  # loop again so Claude can give a final text reply

        # No more tools needed — extract the final text reply
        text_parts = [b.text for b in response.content if b.type == "text"]
        return " ".join(text_parts) if text_parts else "Done."


def speak(text):
    print(f"Aurelis: {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()


def main():
    print("=" * 50)
    print("Aurelis is ready. Hold SPACE to talk, ESC to quit.")
    print("=" * 50)
    while True:
        if keyboard.is_pressed("esc"):
            print("Shutting down.")
            break
        if keyboard.is_pressed("space"):
            audio = record_audio()
            if audio is None:
                continue
            print("Transcribing...")
            text = transcribe(audio)
            if not text:
                print("(didn't catch that)")
                continue
            print(f"You: {text}")
            reply = ask_claude(text)
            speak(reply)
        sd.sleep(50)


if __name__ == "__main__":
    main()