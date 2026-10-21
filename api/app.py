"""MedScriptAI API: one endpoint, three tasks, each routed to its own LoRA adapter inside vLLM."""
import json
import os
import time
import uuid

from fastapi import FastAPI, Header, HTTPException
from langsmith import traceable                   # turns a function call into a LangSmith trace
from langsmith.wrappers import wrap_openai        # also traces every OpenAI-style call inside it
from openai import OpenAI
from pydantic import BaseModel, Field

from common import SYSTEM                         # SAME system prompts as training (no train/serve skew)

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000/v1")
API_KEY = os.getenv("API_KEY", "")                # empty = no auth (local only); set it in production
TASK_TO_MODEL = {                                 # which adapter answers which task (change without code edits)
    "qa": os.getenv("MODEL_QA", "qa-dpo"),
    "notes": os.getenv("MODEL_NOTES", "notes"),
    "summary": os.getenv("MODEL_SUMMARY", "summary"),
}

llm = wrap_openai(OpenAI(base_url=VLLM_URL, api_key="EMPTY", timeout=120))
app = FastAPI(title="MedScriptAI", version="1.0.0")


class GenerateRequest(BaseModel):
    task: str = Field(pattern="^(qa|notes|summary)$", examples=["qa"])
    input: str = Field(min_length=1, max_length=20000)
    max_tokens: int = Field(default=700, ge=1, le=2000)
    temperature: float = Field(default=0.0, ge=0.0, le=1.5)


class GenerateResponse(BaseModel):
    request_id: str
    task: str
    model: str
    output: str
    parsed_json: dict | None = None               # filled for task="summary" when the output is valid JSON
    latency_ms: int
    disclaimer: str = "AI-generated. For education/documentation support only. Not medical advice."


@traceable(name="medscriptai.generate", run_type="chain")
def run_generation(task: str, text: str, max_tokens: int, temperature: float) -> dict:
    model = TASK_TO_MODEL[task]
    resp = llm.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM[task]}, {"role": "user", "content": text}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return {"model": model, "output": resp.choices[0].message.content or ""}


@app.get("/health")
def health():
    """Liveness: the API process is up."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness: vLLM answers and every adapter we route to is loaded."""
    try:
        served = {m.id for m in llm.models.list().data}
    except Exception as exc:
        raise HTTPException(503, f"vLLM not reachable: {exc.__class__.__name__}")
    missing = sorted(set(TASK_TO_MODEL.values()) - served)
    if missing:
        raise HTTPException(503, f"adapters not loaded in vLLM: {missing}")
    return {"status": "ready", "models": sorted(served)}


@app.post("/v1/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest, x_api_key: str = Header(default="")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "invalid or missing X-API-Key header")
    start = time.perf_counter()
    try:
        result = run_generation(req.task, req.input, req.max_tokens, req.temperature)
    except Exception as exc:
        raise HTTPException(502, f"model backend error: {exc.__class__.__name__}")
    parsed = None
    if req.task == "summary":
        try:
            parsed = json.loads(result["output"])
        except json.JSONDecodeError:
            parsed = None
    return GenerateResponse(request_id=str(uuid.uuid4()), task=req.task, model=result["model"],
                            output=result["output"], parsed_json=parsed,
                            latency_ms=int((time.perf_counter() - start) * 1000))
