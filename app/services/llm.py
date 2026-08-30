import os
import json
import re
import logging
from dotenv import load_dotenv
load_dotenv()
from app.services.prompts import SYSTEM_INSTRUCTION, build_user_prompt

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.35"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "900"))
LLM_TIMEOUT_S = int(os.getenv("LLM_TIMEOUT_S", "30"))
LLM_ENABLED = bool(GEMINI_API_KEY) and os.getenv("LLM_DISABLE") != "1"

_client = None
def _get_model():
    global _client
    if _client is None:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        _client = genai.GenerativeModel(GEMINI_MODEL, system_instruction=SYSTEM_INSTRUCTION)
    return _client

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "top_reasons": {"type": "array", "items": {"type": "string"}},
        "recommended_action": {"type": "string"},
        "priority": {"type": "string", "enum": ["P0","P1","P2","P3"]},
        "draft_message": {"type": "string"},
        "playbook_citation": {"type": "string"},
    },
    "required": ["summary","top_reasons","recommended_action","priority","draft_message"],
}

generation_config = {
    "temperature": LLM_TEMPERATURE,
    "response_mime_type": "application/json",
    "response_schema": RESPONSE_SCHEMA,
}

def safe_json_parse(raw: str) -> dict | None:
    # strip markdown code blocks
    cleaned = re.sub(r'^```json\s*', '', raw, flags=re.MULTILINE)
    cleaned = re.sub(r'^```\s*', '', cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Try to salvage between first { and last }
    start = raw.find('{')
    end = raw.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start:end+1])
        except json.JSONDecodeError:
            pass
            
    return None

def _validate_and_coerce(obj: dict, base_action: dict, playbook_snippet: dict) -> dict:
    # ensure required keys
    out = {
        "summary": str(obj.get("summary", "")).strip()[:600],
        "top_reasons": obj.get("top_reasons", []),
        "recommended_action": str(obj.get("recommended_action", "")).strip(),
        "priority": base_action["priority"],  # FORCE
        "draft_message": str(obj.get("draft_message", "")).strip()[:900],
        "playbook_citation": str(obj.get("playbook_citation", "")).strip()
    }
    
    if not out["playbook_citation"]:
        out["playbook_citation"] = playbook_snippet.get("title", "")
        
    if not isinstance(out["top_reasons"], list):
        out["top_reasons"] = []
        
    out["top_reasons"] = [str(r).strip() for r in out["top_reasons"] if str(r).strip()]
    if len(out["top_reasons"]) > 3:
        out["top_reasons"] = out["top_reasons"][:3]
        
    action_label = str(base_action.get("action_label", "")).strip()
    # fuzzy check if the recommended action drifted off-topic
    action_keywords = set(re.findall(r'\w+', action_label.lower()))
    rec_action_keywords = set(re.findall(r'\w+', out["recommended_action"].lower()))
    
    if not action_keywords.intersection(rec_action_keywords) and action_label:
        out["recommended_action"] = f"{action_label} - {out['recommended_action']}"
        
    # strip any markdown / stray backticks from strings
    for k in ["summary", "recommended_action", "draft_message", "playbook_citation"]:
        out[k] = out[k].replace("```", "").strip()
        
    # fallback for draft_message if it's completely missing
    if not out["draft_message"]:
        out["draft_message"] = _template_message({"name": ""}, base_action, playbook_snippet)
    else:
        out["draft_message"] = re.sub(r'\[.*?\]', '', out["draft_message"])
        
    return out

def _template_message(signals, base_action, playbook_snippet):
    name = signals.get('name', 'Customer')
    plan = signals.get('plan_tier', 'your')
    
    # Very basic canned message
    if base_action.get('channel') == 'email':
        return f"Hi {name},\n\nI noticed some changes in your {plan} account activity and wanted to check in. Is there anything we can help you with to ensure you're getting the most value?\n\nBest,\nYour CSM"
    elif base_action.get('channel') == 'call':
        return f"Call script for {name}: 'Hi {name}, calling to check on your {plan} plan usage. We have some great new features that might help your team...'"
    else:
        return f"Checking in with {name} regarding their {plan} plan."

def _fallback(signals, churn_prob, breakdown, playbook_snippet, base_action, reason="disabled"):
    top_contribs = breakdown.get("contributors", [])[:3]
    driver_label = top_contribs[0]['label'] if top_contribs else "overall usage"
    
    summary = (f"{signals.get('name', 'Customer')} ({signals.get('plan_tier', 'Plan')}) shows "
               f"{breakdown.get('risk_band', 'unknown')} risk (health "
               f"{breakdown.get('health_score', 0)}/100, ML churn {churn_prob:.0%}). "
               f"Primary driver: {driver_label}. "
               f"Per playbook '{playbook_snippet.get('title','')}', "
               f"{base_action.get('action_label', '').lower()} is recommended.")
               
    top_reasons = [f"{c['label']}: {c['detail']}" for c in top_contribs]
    
    return {
        "summary": summary,
        "top_reasons": top_reasons,
        "recommended_action": base_action.get("action_label", ""),
        "priority": base_action.get("priority", "P3"),
        "draft_message": _template_message(signals, base_action, playbook_snippet),
        "playbook_citation": playbook_snippet.get("title", ""),
        "_generated_by": "fallback"
    }

def analyze(signals, churn_prob, breakdown, playbook_snippet, base_action) -> dict:
    if not LLM_ENABLED:
        return _fallback(signals, churn_prob, breakdown, playbook_snippet, base_action, reason="disabled")
        
    prompt = build_user_prompt(signals, churn_prob, breakdown, playbook_snippet, base_action)
    
    for attempt in (1, 2):
        try:
            resp = _get_model().generate_content(
                prompt,
                generation_config=generation_config,
                request_options={"timeout": LLM_TIMEOUT_S},
            )
            obj = safe_json_parse(resp.text)
            if obj is None and attempt == 1:
                prompt = prompt + "\n\nYour previous output was not valid JSON. Return ONLY the JSON object, nothing else."
                continue
            if obj is None:
                break
            
            out = _validate_and_coerce(obj, base_action, playbook_snippet)
            out["_generated_by"] = f"gemini:{GEMINI_MODEL}"
            out["_attempt"] = attempt
            return out
        except Exception as e:
            logging.warning("LLM attempt %s failed: %s", attempt, e)
            if attempt == 2:
                break
                
    return _fallback(signals, churn_prob, breakdown, playbook_snippet, base_action, reason="llm_error_or_bad_json")
