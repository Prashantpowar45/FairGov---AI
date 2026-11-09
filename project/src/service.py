import os
import joblib
import pandas as pd
from .rules import ROOT, FEATURES, check
from .rag import retrieve


# ── User-facing English strings — Gemini translates these on request ─────────
_MESSAGES = {
    'route_a_message': (
        'Your application looks good! It has been sent to a government officer '
        'for final approval and processing.'
    ),
    'route_b_message': (
        'Unfortunately your application does not meet the eligibility criteria at this time. '
        'Please read the explanation below to understand what needs to be corrected before reapplying.'
    ),
    'route_b_action': (
        'Please review the reasons listed above, make the necessary corrections, '
        'and resubmit your application once the conditions are met.'
    ),
    'route_c_message': (
        'Your application requires additional verification. '
        'It has been forwarded to a government officer who will review it and reach out to you shortly.'
    ),
}


def _translate(texts: dict, language: str) -> dict:
    """Translate a dict of {key: english_text} into the requested language using Gemini.

    Falls back to English originals if language is English, Gemini is not
    configured, or the API call fails.

    Args:
        texts:    {key: english_string, ...}
        language: 'English' | 'Marathi' | 'Hindi'

    Returns:
        {key: translated_string, ...}
    """
    if language == 'English':
        return texts

    api_key = os.getenv('GEMINI_API_KEY', '').strip()
    if not api_key:
        return texts  # Gemini not configured — return English fallback

    try:
        import json as _json
        from google import genai

        numbered = '\n'.join(f'{i + 1}. {v}' for i, v in enumerate(texts.values()))
        prompt = (
            f'Translate each numbered sentence below into {language}. '
            'Keep the tone simple, friendly, and suitable for a student applying '
            'for a government scholarship. '
            'Return ONLY a JSON array of translated strings in the same order, '
            'with no extra text.\n\n'
            f'{numbered}'
        )
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=os.getenv('GEMINI_MODEL', 'gemini-2.5-flash'),
            contents=prompt,
        )
        raw = (response.text or '').strip()
        # Strip markdown code fences if Gemini wraps in ```json ... ```
        if raw.startswith('```'):
            raw = '\n'.join(raw.splitlines()[1:])
        if raw.endswith('```'):
            raw = raw.rsplit('```', 1)[0]
        translated_list = _json.loads(raw.strip())
        return dict(zip(texts.keys(), translated_list))
    except Exception:
        return texts  # any error → safe English fallback


def predict(student, explain=False):
    """Run XGBoost prediction + deterministic rule check.

    Args:
        student: dict with the 12 FEATURES keys.
        explain: if True, also compute SHAP feature contributions.

    Returns:
        dict with ml_prediction, ml_probability_eligible, rule_engine,
        mismatch flag, and optionally shap dict.
    """
    model = joblib.load(ROOT / 'models/xgboost_dte.joblib')
    X = pd.DataFrame([{key: student[key] for key in FEATURES}])[FEATURES]
    y = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])
    policy = check(student)
    result = {
        'ml_prediction': bool(y),
        'ml_probability_eligible': round(prob, 5),
        'rule_engine': policy,
        'mismatch': bool(y) != policy['preliminary_eligible'],
        'disclaimer': 'Demonstration prediction only; not an official scholarship decision.',
    }
    if explain:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = explainer(X)
        vals = sv.values[0]
        result['shap'] = {
            'base_value': float(sv.base_values[0]),
            'feature_contributions': {
                k: round(float(v), 5) for k, v in zip(FEATURES, vals)
            },
            'note': 'SHAP values explain model scores, not policy/legal reasons.',
        }
    return result


def evaluate(student, language='English', top_k=4):
    """Full 3-path decision router.

    Route A — Both ML and rules say eligible:
        Forward to government employee for final approval.

    Route B — Both ML and rules say not eligible:
        Run SHAP + RAG to produce a cited, policy-grounded rejection
        explanation that the student can act on to reapply.

    Route C — ML and rules disagree (mismatch):
        Escalate to government employee for manual review.

    All user-facing message strings are translated into the requested language
    via Gemini. Falls back to English if Gemini is not configured.

    Args:
        student:  dict with the 12 FEATURES keys.
        language: 'English' | 'Marathi' | 'Hindi'
        top_k:    number of policy documents to retrieve for Route B.

    Returns:
        dict with 'route' key ('route_a' | 'route_b' | 'route_c') and
        route-specific payload (see AGENT.md § 6 for full schema).
    """
    raw = predict(student, explain=True)

    ml_eligible = raw['ml_prediction']
    rule_eligible = raw['rule_engine']['preliminary_eligible']
    mismatch = raw['mismatch']

    base = {
        'ml_prediction': ml_eligible,
        'ml_probability_eligible': raw['ml_probability_eligible'],
        'rule_engine': raw['rule_engine'],
        'mismatch': mismatch,
        'disclaimer': raw['disclaimer'],
    }

    # ── Route A: Both eligible ──────────────────────────────────────────────
    if ml_eligible and rule_eligible:
        t = _translate({'message': _MESSAGES['route_a_message']}, language)
        return {
            **base,
            'route': 'route_a',
            'decision': 'eligible',
            'forward_to_employee': True,
            'message': t['message'],
        }

    # ── Route C: Mismatch — ML and rules disagree ───────────────────────────
    if mismatch:
        t = _translate({'message': _MESSAGES['route_c_message']}, language)
        return {
            **base,
            'route': 'route_c',
            'decision': 'escalated',
            'forward_to_employee': True,
            'message': t['message'],
        }

    # ── Route B: Both rejected — SHAP + RAG explanation ────────────────────
    failed_rules = raw['rule_engine']['failed_rules']
    shap_contributions = raw['shap']['feature_contributions']

    # Top contributing features that pushed the score DOWN (most negative first)
    top_negative = sorted(
        ((f, v) for f, v in shap_contributions.items() if v < 0),
        key=lambda x: x[1],
    )[:3]

    # Build a focused policy query from failed rules + SHAP-flagged features
    query_terms = failed_rules + [f for f, _ in top_negative]
    if query_terms:
        query = 'Eligibility requirements and conditions for: ' + ', '.join(query_terms)
    else:
        query = 'DTE Maharashtra scholarship eligibility criteria and disqualifying conditions'

    from .rag_full import answer as rag_answer
    explanation = rag_answer(query, top_k, language)

    # Translate all fixed user-facing strings for Route B
    t = _translate(
        {
            'message': _MESSAGES['route_b_message'],
            'recommended_action': _MESSAGES['route_b_action'],
        },
        language,
    )

    return {
        **base,
        'route': 'route_b',
        'decision': 'rejected',
        'forward_to_employee': False,
        'failed_rules': failed_rules,
        'shap_top_factors': [
            {'feature': f, 'contribution': round(v, 5)} for f, v in top_negative
        ],
        'policy_explanation': explanation.get(
            'answer', 'See policy_hits for relevant clauses.'
        ),
        'policy_hits': explanation.get('hits', []),
        'answer_mode': explanation.get('answer_mode', 'retrieval_only'),
        'recommended_action': t['recommended_action'],
        'message': t['message'],
    }
