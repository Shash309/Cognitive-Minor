import math

def get_static_questions():
    """
    Returns the static questions for Phase 1 of the AI Career Quiz.
    """
    return [
        {
            "id": "Q1",
            "type": "scale",
            "question": "Rate your experience (1-5):",
            "options": [
                "Programming / Coding",
                "Mathematics / Logical Problem Solving",
                "Writing / Communication",
                "Design (UI/UX, creative tools)",
                "Business / Finance",
                "Science (Physics/Chem/Bio)"
            ],
            "scale_range": [1, 5]
        },
        {
            "id": "Q2",
            "type": "multi",
            "question": "What have you actually done?",
            "options": [
                "Built projects",
                "Participated in competitions",
                "Led a team",
                "Internships / real-world work",
                "None"
            ]
        },
        {
            "id": "Q3",
            "type": "multi",
            "question": "Which problems do you enjoy most? (Pick max 2)",
            "options": [
                "Analytical",
                "Creative",
                "Human-centric",
                "Strategic",
                "Scientific"
            ]
        },
        {
            "id": "Q4",
            "type": "single",
            "question": "Work preference:",
            "options": [
                "Build things",
                "Analyze deeply",
                "Manage people",
                "Research/explore"
            ]
        },
        {
            "id": "Q7",
            "type": "single",
            "question": "Uncertainty tolerance:",
            "options": [
                "High",
                "Medium",
                "Low"
            ]
        }
    ]

def _parse_context(context: dict):
    phase1 = context.get("phase1_answers") or {}
    voice = context.get("voice_analysis") or {}
    psych = context.get("psychological_traits") or {}

    def safe_parse(d, k1, k2=None, default=0.5):
        val = d.get(k1)
        if val is None and k2: val = d.get(k2)
        if val is None: return default
        try:
            val_str = str(val).replace('%', '').strip()
            f = float(val_str)
            return f / 100.0 if f > 1.0 else f
        except (ValueError, TypeError):
            return default

    v_conf = safe_parse(voice, "confidence", "confidence_score", 0.5)
    v_hes  = safe_parse(voice, "hesitation", None, 0.5)
    
    p_open = safe_parse(psych, "openness", None, 0.5)
    p_cons = safe_parse(psych, "conscientiousness", None, 0.5)
    p_ext  = safe_parse(psych, "extraversion", None, 0.5)
    p_agr  = safe_parse(psych, "agreeableness", None, 0.5)
    p_neu  = safe_parse(psych, "neuroticism", None, 0.5)

    # Baseline logical/creative/social signals from Phase 1
    base_logical = 0.5
    base_creative = 0.5
    base_social = 0.5
    base_risk = 0.5

    q1 = phase1.get("Q1", {})
    if isinstance(q1, dict):
        base_logical += (q1.get("Programming / Coding", 1) + q1.get("Mathematics / Logical Problem Solving", 1)) * 0.05
        base_creative += q1.get("Design (UI/UX, creative tools)", 1) * 0.1
        base_social += q1.get("Writing / Communication", 1) * 0.1

    q3 = phase1.get("Q3", [])
    if isinstance(q3, list):
        if "Analytical" in q3: base_logical += 0.2
        if "Creative" in q3: base_creative += 0.2
        if "Human-centric" in q3: base_social += 0.2
    
    q4 = phase1.get("Q4")
    if q4 == "Build things": base_logical += 0.1
    if q4 == "Manage people": base_social += 0.2
    
    q7 = phase1.get("Q7")
    if q7 == "High": base_risk += 0.3
    if q7 == "Low": base_risk -= 0.2

    return {
        "logical": min(1.0, base_logical),
        "creative": min(1.0, base_creative),
        "social": min(1.0, base_social),
        "risk": min(1.0, max(0.0, base_risk)),
        "voice": {"conf": v_conf, "hes": v_hes},
        "psych": {"open": p_open, "cons": p_cons, "ext": p_ext, "neu": p_neu}
    }


def get_adaptive_questions(context: dict) -> list:
    """
    Phase 2: Adaptive Question Generation
    Rules: Generate 5-10 adaptive questions based on Phase 1, Voice, Psych
    """
    parsed = _parse_context(context)
    questions = []
    idx = 101

    # 1. Logical Inclination
    if parsed["logical"] > 0.6:
        questions.extend([
            {"id": f"A_{idx}", "type": "single", "question": "In a technical project, do you prefer:", "options": ["Building from scratch", "Debugging and optimizing", "Designing the system architecture"]},
            {"id": f"A_{idx+1}", "type": "single", "question": "When solving a complex problem, you:", "options": ["Break it down systematically", "Look for an existing pattern", "Experiment until it works"]}
        ])
        idx += 2

    # 2. Creativity Detection
    if parsed["creative"] > 0.6 or parsed["psych"]["open"] > 0.7:
        questions.extend([
            {"id": f"A_{idx}", "type": "single", "question": "How often do you generate new concepts?", "options": ["Constantly", "When required", "Rarely"]},
            {"id": f"A_{idx+1}", "type": "single", "question": "Would you rather:", "options": ["Innovate completely new ideas", "Optimize existing structures"]}
        ])
        idx += 2

    # 3. Social Orientation
    if parsed["psych"]["ext"] > 0.6 or parsed["social"] > 0.6:
        questions.extend([
            {"id": f"A_{idx}", "type": "single", "question": "In a team, do you prefer to:", "options": ["Lead and delegate", "Collaborate equally", "Work as an individual contributor"]},
            {"id": f"A_{idx+1}", "type": "single", "question": "Does social interaction:", "options": ["Energize you", "Neutral", "Drain you"]}
        ])
        idx += 2
    elif parsed["psych"]["ext"] < 0.4:
        questions.extend([
            {"id": f"A_{idx}", "type": "single", "question": "For deep work, you prefer:", "options": ["Total isolation", "A quiet shared space", "Headphones in a cafe"]},
            {"id": f"A_{idx+1}", "type": "single", "question": "When faced with a blocker, you:", "options": ["Solve it independently", "Ask for help immediately", "Research online before asking"]}
        ])
        idx += 2

    # 4. Voice-Based Adaptation
    v_conf = parsed["voice"]["conf"]
    if v_conf < 0.4:
        questions.extend([
            {"id": f"A_{idx}", "type": "single", "question": "When making major decisions, what holds you back?", "options": ["Fear of failure", "Lack of information", "Overthinking"]},
            {"id": f"A_{idx+1}", "type": "single", "question": "How do you handle self-doubt?", "options": ["Talk it out", "Push through it", "Step back and reassess"]}
        ])
        idx += 2
    elif v_conf > 0.7:
        questions.extend([
            {"id": f"A_{idx}", "type": "single", "question": "Are you comfortable taking calculated risks?", "options": ["Absolutely", "If the data supports it", "Only forced to"]},
            {"id": f"A_{idx+1}", "type": "single", "question": "How do you lead?", "options": ["By example", "By strategic delegation", "I prefer being an independent ace"]}
        ])
        idx += 2

    # 5. Risk Profile
    if parsed["risk"] < 0.4 or parsed["psych"]["cons"] > 0.7:
        questions.extend([
            {"id": f"A_{idx}", "type": "single", "question": "You value:", "options": ["Structured environments", "Flexible chaos"]},
            {"id": f"A_{idx+1}", "type": "single", "question": "When planning your career, you look:", "options": ["10 years ahead", "2-3 years ahead", "I improvise"]}
        ])
        idx += 2
    elif parsed["risk"] > 0.6:
        questions.extend([
            {"id": f"A_{idx}", "type": "single", "question": "Startups vs Corporations:", "options": ["High ownership startup", "Stable corporation"]},
            {"id": f"A_{idx+1}", "type": "single", "question": "Are you driven by:", "options": ["High upside/equity", "Predictable growth"]}
        ])
        idx += 2

    # Cap to max 10 adaptive questions
    return questions[:10]


def extract_core_scores(context: dict, adaptive_answers: dict) -> dict:
    parsed = _parse_context(context)
    
    # Base from parsed logic
    logical = parsed["logical"]
    creativity = parsed["creative"]
    social = parsed["social"]
    risk = parsed["risk"]
    leadership = 0.5
    stability = 0.5
    
    # Adjust using Psych & Voice
    # voice confidence -> leadership & risk
    leadership += (parsed["voice"]["conf"] - 0.5) * 0.4
    risk += (parsed["voice"]["conf"] - 0.5) * 0.2
    
    # hesitation -> reduces confidence weight
    leadership -= parsed["voice"]["hes"] * 0.2
    
    # openness -> boosts creativity
    creativity += (parsed["psych"]["open"] - 0.5) * 0.3
    
    # conscientiousness -> boosts stability
    stability += (parsed["psych"]["cons"] - 0.5) * 0.4
    
    # Refine using adaptive answers
    for qid, ans in adaptive_answers.items():
        if isinstance(ans, list) and len(ans) > 0: ans = ans[0]
        if not ans: continue
        
        if ans in ["Designing the system architecture", "System building"]: logical += 0.1; leadership += 0.1
        if ans in ["Innovate completely new ideas", "Constantly"]: creativity += 0.2; risk += 0.1
        if ans in ["Lead and delegate", "High ownership startup"]: leadership += 0.2; risk += 0.2
        if ans in ["Total isolation", "Solve it independently"]: social -= 0.15; logical += 0.1
        if ans in ["Fear of failure", "Overthinking", "Structured environments", "Stable corporation"]: risk -= 0.2; stability += 0.2
        if ans in ["Absolutely", "High upside/equity"]: risk += 0.2; stability -= 0.1

    # Normalize to 0-1
    return {
        "logical": min(1.0, max(0.0, logical)),
        "creativity": min(1.0, max(0.0, creativity)),
        "social": min(1.0, max(0.0, social)),
        "leadership": min(1.0, max(0.0, leadership)),
        "risk": min(1.0, max(0.0, risk)),
        "stability": min(1.0, max(0.0, stability))
    }

def match_careers(scores: dict) -> list:
    clusters = {
        "Software Engineer": {"logical": 0.9, "creativity": 0.4, "social": 0.2, "leadership": 0.3, "risk": 0.4, "stability": 0.7},
        "Product Manager":   {"logical": 0.6, "creativity": 0.6, "social": 0.9, "leadership": 0.9, "risk": 0.6, "stability": 0.5},
        "Designer":          {"logical": 0.3, "creativity": 0.9, "social": 0.6, "leadership": 0.4, "risk": 0.5, "stability": 0.5},
        "Entrepreneur":      {"logical": 0.5, "creativity": 0.8, "social": 0.8, "leadership": 0.9, "risk": 0.9, "stability": 0.1},
        "Data Scientist":    {"logical": 0.9, "creativity": 0.5, "social": 0.3, "leadership": 0.4, "risk": 0.4, "stability": 0.6},
        "Consultant":        {"logical": 0.7, "creativity": 0.5, "social": 0.9, "leadership": 0.7, "risk": 0.6, "stability": 0.4},
        "Researcher":        {"logical": 0.9, "creativity": 0.6, "social": 0.2, "leadership": 0.3, "risk": 0.2, "stability": 0.8}
    }
    
    results = []
    for career, reqs in clusters.items():
        # Cosine-like or strictly distance based
        dist = 0
        for k in scores.keys():
            dist += abs(scores[k] - reqs.get(k, 0.5)) ** 2
        
        prob = max(0.0, 1.0 - math.sqrt(dist / 6.0))
        results.append({"career": career, "probability": prob})
        
    results.sort(key=lambda x: x["probability"], reverse=True)
    return results

def process_career_recommendation(context: dict, adaptive_answers: dict) -> dict:
    """
    Final Output Generation matches strict formatting.
    """
    scores = extract_core_scores(context, adaptive_answers)
    career_list = match_careers(scores)
    
    # Generate deep reasoning
    top = career_list[0]
    reasoning = f"Based on your high alignment with {top['career']} ({round(top['probability']*100)}%), this fits your profile."
    strengths = []
    weaknesses = []
    
    if scores["logical"] > 0.7: strengths.append("Strong analytical and problem-solving pipeline.")
    if scores["creativity"] > 0.7: strengths.append("High divergent thinking and design skills.")
    if scores["leadership"] > 0.7: strengths.append("Natural leadership and clear delegation.")
    if scores["social"] < 0.4: weaknesses.append("May struggle in highly extroverted cross-functional roles.")
    if scores["risk"] < 0.3: weaknesses.append("Can be overly cautious in rapid-change environments.")
    
    if not strengths: strengths = ["Balanced versatile skill set."]
    if not weaknesses: weaknesses = ["Generalist profile may lack deep specialization."]
    
    p_sum = "A balanced analytical mind" if scores["logical"] > scores["creativity"] else "A highly creative and flexible thinker"
    if scores["risk"] > 0.7: p_sum += " who thrives on risk and entrepreneurship."
    elif scores["stability"] > 0.7: p_sum += " who values structured, long-term impact."
    else: p_sum += " with a versatile approach to problem solving."
    
    v_conf = context.get("voice_analysis", {}).get("confidence", 0.5)
    if v_conf > 0.7:
        c_an = "Voice analysis indicates high confidence and clarity, reinforcing leadership potentials."
    elif v_conf < 0.4:
        c_an = "Voice patterns suggest some hesitation; leaning towards roles providing psychological safety and structure."
    else:
        c_an = "Voice metrics reflect a balanced and measured communication style."
        
    return {
        "scores": scores,
        "career_recommendations": career_list,
        "top_careers": [{"name": c["career"], "score": c["probability"] * 100} for c in career_list[:3]],
        "insights": {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "personality_summary": p_sum,
            "confidence_analysis": c_an
        },
        "confidence_level": "HIGH" if top["probability"] > 0.8 else "MEDIUM" if top["probability"] > 0.6 else "LOW",
        "reasoning": [reasoning, p_sum, c_an]
    }
