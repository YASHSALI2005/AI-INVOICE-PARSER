
import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

api_key = os.getenv("CLAUDE_API_KEY")

if not api_key:
    print("Error: CLAUDE_API_KEY not found.")
    exit(1)

client = anthropic.Anthropic(api_key=api_key)

models_to_test = [
    "claude-3-5-sonnet-20241022", # New Sonnet 3.5 (Oct)
    "claude-3-5-sonnet-20240620", # Original Sonnet 3.5
    "claude-3-5-sonnet-latest",   # Alias
    "claude-3-opus-20240229",     # Opus
    "claude-3-sonnet-20240229",   # Sonnet 3
    "claude-3-haiku-20240307",    # Haiku
    "claude-2.1",                 # Legacy
]

print(f"Testing {len(models_to_test)} models with key: {api_key[:15]}...")

for model in models_to_test:
    try:
        print(f"Testing {model:<30} ...", end=" ", flush=True)
        client.messages.create(
            model=model,
            max_tokens=10,
            messages=[{"role": "user", "content": "Hi"}]
        )
        print("SUCCESS ✅")
    except anthropic.NotFoundError:
        print("NOT FOUND ❌")
    except anthropic.AuthenticationError:
        print("AUTH ERROR 🚫")
    except anthropic.BadRequestError as e:
         print(f"BAD REQUEST ⚠️ ({e.message})")
    except Exception as e:
        print(f"ERROR 💥 {str(e)}")
