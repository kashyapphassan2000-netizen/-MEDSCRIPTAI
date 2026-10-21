"""A fake OpenAI-compatible server (like vLLM) for testing pipelines and the API WITHOUT a GPU.

Run:  uvicorn tests.mock_openai:app --port 8000
It answers /v1/models and /v1/chat/completions with small, predictable replies.
"""
import re
import time
from fastapi import FastAPI, Request

app = FastAPI()
MODELS = ["base", "qa", "notes", "summary", "qa-dpo", "notes-dpo", "teacher"]


def reply_for(model, text):
    """Pick a fake answer that has the same SHAPE as a real one."""
    n_texts = len(re.findall(r"<text \d+>", text))
    if n_texts:                                            # UltraFeedback judge request
        blocks = [f"#### Output for Text {i + 1}\nRating: {5 - i if i < 4 else 1}\nRationale: mock rationale."
                  for i in range(n_texts)]
        return "\n\n".join(blocks)
    if "The correct answer is:" in text:                   # teacher explanation request (Part 5)
        gold = text.split("The correct answer is:", 1)[1].splitlines()[0].strip()
        return (f"{gold} is correct because it best fits the key findings in the stem. "
                "The other options do not explain the presentation as well. (mock teacher)")
    if '"score": <1-5>' in text:                           # 32_judge.py --mode score
        return '{"score": 4, "reason": "mock grade"}'
    if '"winner"' in text:                                 # 32_judge.py --mode pairwise
        answer_a = text.split("ANSWER A:", 1)[1].split("ANSWER B:", 1)[0]
        return '{"winner": "%s", "reason": "mock"}' % ("A" if "dpo" in answer_a else "B")
    if "Return ONLY valid JSON" in text:                   # summary adapter
        return '{"visit motivation": "mock", "patient information": {"age": "40"}}'
    if "clinical case note" in text:                       # notes adapter
        return "A 40-year-old patient presented with mock symptoms."
    letter = "B" if model.startswith("qa") else "A"        # qa adapters / base / teacher
    return f"Answer: {letter}) mock option\nExplanation: mock explanation from {model}."


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": m, "object": "model"} for m in MODELS]}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    text = "\n".join(str(m.get("content", "")) for m in body.get("messages", []))
    content = reply_for(body.get("model", "base"), text)
    return {
        "id": "mock-1", "object": "chat.completion", "created": int(time.time()), "model": body.get("model"),
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": len(text) // 4, "completion_tokens": len(content) // 4,
                  "total_tokens": (len(text) + len(content)) // 4},
    }
