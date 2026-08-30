import json

SYSTEM_INSTRUCTION = """
You are a Customer Success retention analyst. You receive structured signals,
an ML churn probability, a deterministic risk breakdown, one retrieved
retention-playbook snippet, and a REQUIRED action already chosen by the
system. Your job:
  - Explain WHY this customer is at risk, grounding every reason in the
    provided signals or risk breakdown. Never invent numbers or events.
  - Produce a short, specific outreach message the CSM can send with minimal
    edits, personalized to this customer's situation and plan.
  - Keep `recommended_action` as an elaboration of the REQUIRED action -
    do NOT substitute a different action. Keep `priority` exactly as given.
  - Reference the playbook snippet explicitly in `summary`
    (e.g. "Per playbook 'Repeated payment failures', ...").
Return ONLY the JSON object matching the schema. No markdown, no prose outside
JSON.
"""

def build_user_prompt(signals, churn_prob, breakdown, playbook_snippet, base_action) -> str:
    compact = {
      "name": signals.get("name"),
      "plan_tier": signals.get("plan_tier"),
      "tenure_days": signals.get("tenure_days"),
      "arr": signals.get("arr"),
      "days_to_renewal": signals.get("days_to_renewal"),
      "usage_trend_pct": signals.get("usage_trend_pct"),
      "last_login_days": signals.get("last_login_days"),
      "logins_last_30": signals.get("logins_last_30"),
      "payment_failures": signals.get("payment_failures"),
      "support_contacts": signals.get("support_contacts"),
      "open_tickets": signals.get("open_tickets"),
      "avg_sentiment": signals.get("avg_sentiment"),
      "engagement_score": signals.get("engagement_score"),
    }
    
    return f"""
CUSTOMER SIGNALS:
{json.dumps(compact, indent=2, default=str)}

ML CHURN PROBABILITY: {churn_prob:.0%}
ML TOP DRIVERS: {json.dumps(signals.get("ml_top_features", []), default=str)}

DETERMINISTIC RISK BREAKDOWN (health {breakdown.get('health_score')}/100,
band {breakdown.get('risk_band')}):
{json.dumps(breakdown.get('contributors', []), indent=2, default=str)}
Positives: {json.dumps(breakdown.get('positives', []), default=str)}

RETRIEVED PLAYBOOK SNIPPET:
  title: {playbook_snippet.get('title')}
  guidance: {playbook_snippet.get('text')}
  (retrieval_method: {playbook_snippet.get('retrieval_method')},
   low_confidence: {playbook_snippet.get('low_confidence')})

REQUIRED ACTION (do not change):
  action_label: {base_action.get('action_label')}
  priority: {base_action.get('priority')}
  channel: {base_action.get('channel')}
  rationale: {base_action.get('rationale')}

TASK:
  1. summary: 2-3 sentences. Must cite the playbook title. State the single
     biggest risk driver and the ML probability in plain language.
  2. top_reasons: up to 3 bullet strings, each tied to a specific signal
     value above (quote the number).
  3. recommended_action: restate REQUIRED action_label, elaborated into a
     concrete next step for the CSM via {base_action.get('channel')}.
  4. priority: exactly "{base_action.get('priority')}".
  5. draft_message: 60-120 words, addressed to {signals.get('name')},
     appropriate to a {signals.get('plan_tier')} plan, matching the playbook
     guidance and channel tone. Do NOT use any [NAME] or [Company] placeholders.
     Sign off with "Your Customer Success Team".
  6. playbook_citation: the playbook title you used.
If the playbook snippet is low_confidence or its guidance conflicts with the
REQUIRED action, prefer the REQUIRED action and keep the playbook for tone
only; still fill playbook_citation.
"""
