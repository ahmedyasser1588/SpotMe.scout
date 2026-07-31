import os
import re
import json
import time
import logging
import tempfile
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from groq import Groq
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER

# إعداد مسار المشروع لضمان الاستيراد التلقائي في بيئة Vercel Serverless
file_path = Path(__file__).resolve()
sys.path.append(str(file_path.parent))

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_RETRIES = 3
BASE_RETRY_DELAY_SECONDS = 1.5
DEFAULT_TEMPERATURE = 0.3
MAX_INTERVIEW_QUESTIONS = 25
MAX_FOLLOWUP_ROUNDS = 5
MAX_REVERIFICATION_PASSES = 2

# تعديل مسار الحفظ ليكون داخل مجلد /tmp المسموح به في Vercel Serverless
OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "spotme_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SUPPORTED_SPORTS = ["Football", "Basketball", "Volleyball", "Handball"]

SPORT_FOCUS_AREAS = {
    "Football": ["position", "preferred_foot", "league_level", "playing_style"],
    "Basketball": ["position", "shooting_style", "dominant_hand", "competition_level"],
    "Volleyball": ["position", "blocking_role", "attack_role", "serving_style"],
    "Handball": ["position", "dominant_hand", "offensive_defensive_role"],
}

INTERVIEW_REQUIRED_TOPICS = [
    "name", "age", "country", "position", "dominant_hand_or_foot",
    "height", "weight", "years_of_experience", "training_days_per_week",
    "current_team", "previous_teams", "competitions", "achievements",
    "certificates", "injuries", "coach_name", "career_goal",
    "self reported strengths", "self reported weaknesses",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("spotme_ai_scout")

_groq_client = None

def get_groq_client():
    global _groq_client
    if _groq_client is not None:
        return _groq_client
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key or api_key == "YOUR_GROQ_API_KEY_HERE":
        raise RuntimeError("GROQ_API_KEY environment variable is missing in Vercel settings.")
    _groq_client = Groq(api_key=api_key)
    return _groq_client

def sanitize_filename(name, fallback="player"):
    cleaned = (name or fallback).strip()
    cleaned = re.sub(r"[^A-Za-z0-9_\-]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or fallback

def clamp(value, low, high):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return low
    return max(low, min(high, numeric))

class LLMCallError(Exception):
    pass

def _strip_code_fences(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    return cleaned.strip()

def call_llm(system_prompt, user_prompt, json_mode=True, temperature=DEFAULT_TEMPERATURE, max_tokens=2000):
    client = get_groq_client()
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            kwargs = {}
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise LLMCallError("Empty response received from LLM.")
            return content.strip()
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                delay = BASE_RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
                logger.warning("LLM call failed (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)
                time.sleep(delay)
            else:
                raise LLMCallError(f"LLM call failed after {MAX_RETRIES} attempts: {exc}") from exc
    raise LLMCallError(f"LLM call failed: {last_error}")

def call_llm_json(system_prompt, user_prompt, temperature=DEFAULT_TEMPERATURE, max_tokens=2000):
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = call_llm(system_prompt, user_prompt, json_mode=True, temperature=temperature, max_tokens=max_tokens)
            cleaned = _strip_code_fences(raw)
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict):
                raise ValueError("Parsed JSON is not an object.")
            return parsed
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                logger.warning("JSON parse failed (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)
                time.sleep(BASE_RETRY_DELAY_SECONDS)
                continue
            raise LLMCallError(f"Failed to parse valid JSON after {MAX_RETRIES} attempts: {exc}") from exc
    raise LLMCallError(f"Failed to get valid JSON: {last_error}")

class PlayerProfile(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    country: Optional[str] = None
    sport: Optional[str] = None
    position: Optional[str] = None
    dominant_hand_or_foot: Optional[str] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    years_of_experience: Optional[float] = None
    training_days_per_week: Optional[int] = None
    current_team: Optional[str] = None
    previous_teams: List[str] = Field(default_factory=list)
    competitions: List[str] = Field(default_factory=list)
    achievements: List[str] = Field(default_factory=list)
    certificates: List[str] = Field(default_factory=list)
    injuries: List[str] = Field(default_factory=list)
    coach_name: Optional[str] = None
    career_goal: Optional[str] = None
    strengths_self_reported: List[str] = Field(default_factory=list)
    weaknesses_self_reported: List[str] = Field(default_factory=list)
    sport_specific_attributes: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "ignore"

class VerificationResult(BaseModel):
    corrected_profile: Dict[str, Any]
    confidence_score: float
    warnings: List[str] = Field(default_factory=list)
    missing_fields: List[str] = Field(default_factory=list)

class AgentReport(BaseModel):
    agent_name: str
    score: float
    summary: str
    details: Dict[str, Any] = Field(default_factory=dict)
    evidence: List[str] = Field(default_factory=list)

class HeadScoutReport(BaseModel):
    overall_score: float
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    summary: str
    recommendation: str
    development_plan: List[str] = Field(default_factory=list)

class ExplainabilityReport(BaseModel):
    explanations: Dict[str, str] = Field(default_factory=dict)

def safe_input(prompt, retries=2):
    # تم إلغاء المدخلات التفاعلية كلياً لضمان بيئة الـ Serverless
    return "N/A"

def build_interview_system_prompt(sport):
    focus = SPORT_FOCUS_AREAS.get(sport, [])
    return (
        f"You are an expert sports scouting interviewer specialized in {sport}.\n"
        "Conduct a natural, friendly, ONE-question-at-a-time conversation with an athlete.\n"
        "Never ask a fixed questionnaire; adapt each next question based on the athlete's "
        "previous answers.\n"
        f"Over the course of the interview you must gather: {', '.join(INTERVIEW_REQUIRED_TOPICS)}.\n"
        f"You must also gather these {sport}-specific details: {', '.join(focus)}.\n"
        f"NEVER ask questions relevant to a sport other than {sport}.\n"
        "Ask short, clear, single questions. Do not ask more than one question in the same turn.\n"
        "When you judge that enough information has been gathered to build a complete "
        "player profile, set \"done\" to true and \"next_question\" to null.\n"
        "Respond ONLY with a JSON object of the exact form:\n"
        "{\"next_question\": \"<question text or null>\", \"done\": <true|false>, "
        "\"reasoning\": \"<brief internal reasoning, not shown to the athlete>\"}"
    )

def run_smart_interview(sport, max_questions=MAX_INTERVIEW_QUESTIONS):
    if sport not in SUPPORTED_SPORTS:
        raise ValueError(f"Unsupported sport '{sport}'. Must be one of {SUPPORTED_SPORTS}")

    system_prompt = build_interview_system_prompt(sport)
    conversation = []
    question_count = 0

    logger.info("Starting %s interview.", sport)

    while question_count < max_questions:
        if conversation:
            history_text = "\n".join(
                f"Q: {turn['question']}\nA: {turn['answer']}" for turn in conversation
            )
        else:
            history_text = (
                "No conversation yet. Ask the very first question "
                "(start by asking the athlete's name)."
            )

        user_prompt = (
            f"Conversation so far:\n{history_text}\n\n"
            "Decide the single next question to ask, or mark the interview as done."
        )

        decision = call_llm_json(system_prompt, user_prompt, temperature=0.5, max_tokens=300)
        done = bool(decision.get("done", False))
        next_question = decision.get("next_question")

        if done or not next_question:
            logger.info("Interview complete after %d questions.", question_count)
            break

        # الاستجابة التلقائية في واجهة البرمجة (API)
        answer = "Default response for automated API flow"
        conversation.append({"question": next_question, "answer": answer})
        question_count += 1

    return conversation

def build_player_profile(conversation, sport):
    transcript = "\n".join(f"Q: {t['question']}\nA: {t['answer']}" for t in conversation)
    focus_fields = ", ".join(SPORT_FOCUS_AREAS.get(sport, []))

    system_prompt = (
        "You are a data extraction specialist for sports scouting.\n"
        f"Convert the interview transcript into a structured JSON player profile for {sport}.\n"
        "Use EXACTLY these keys: name, age, country, sport, position, dominant_hand_or_foot, "
        "height_cm, weight_kg, years_of_experience, training_days_per_week, current_team, "
        "previous_teams, competitions, achievements, certificates, injuries, coach_name, "
        "career_goal, strengths_self_reported, weaknesses_self_reported, sport_specific_attributes.\n"
        "previous_teams, competitions, achievements, certificates, injuries, "
        "strengths_self_reported, weaknesses_self_reported must be JSON arrays of strings "
        "(empty array if none mentioned).\n"
        f"sport_specific_attributes must be a JSON object covering: {focus_fields}.\n"
        "Use null for any field not mentioned in the transcript. Never invent data.\n"
        f"Set \"sport\" to \"{sport}\".\n"
        "Respond ONLY with the JSON object, no extra commentary."
    )
    user_prompt = f"Interview transcript:\n{transcript}"

    data = call_llm_json(system_prompt, user_prompt, temperature=0.2, max_tokens=1500)
    data["sport"] = sport

    try:
        profile = PlayerProfile(**data)
    except Exception as exc:
        logger.warning("Profile validation issue, using best-effort profile: %s", exc)
        safe_data = {k: v for k, v in data.items() if k in PlayerProfile.model_fields}
        profile = PlayerProfile(**safe_data)

    return profile

def verify_profile(profile, sport):
    system_prompt = (
        "You are a strict data verification agent for sports scouting profiles.\n"
        "Given a player profile JSON, detect: missing information, inconsistent answers, "
        "impossible values (e.g. negative or unrealistic age, height/weight outside human "
        "ranges, training_days_per_week > 7), and formatting problems.\n"
        "Correct obvious formatting issues (capitalization, numeric types, trimming "
        "whitespace) WITHOUT inventing new factual data.\n"
        "List every field with a null/empty value that is IMPORTANT for a scouting "
        "evaluation as a missing field.\n"
        "Respond ONLY with a JSON object of the exact form:\n"
        "{\"corrected_profile\": {<all original fields, corrected>}, "
        "\"confidence_score\": <float between 0 and 1>, "
        "\"warnings\": [<string>, ...], \"missing_fields\": [<string>, ...]}"
    )
    user_prompt = f"Sport: {sport}\nProfile:\n{profile.model_dump_json()}"

    result = call_llm_json(system_prompt, user_prompt, temperature=0.1, max_tokens=1500)
    result["confidence_score"] = clamp(result.get("confidence_score", 0.0), 0.0, 1.0)
    return VerificationResult(**result)

def resolve_missing_information(profile, verification, sport, max_rounds=MAX_FOLLOWUP_ROUNDS):
    current_profile_dict = dict(verification.corrected_profile)
    missing = list(verification.missing_fields)
    rounds = 0

    while missing and rounds < max_rounds:
        system_prompt = (
            f"You are a follow-up interviewer for {sport} scouting.\n"
            "Given a list of missing profile fields, generate ONE natural follow-up "
            "question that asks for the single most important missing field first.\n"
            "Respond ONLY with JSON: {\"field\": \"<field_name>\", \"question\": \"<question text>\"}"
        )
        user_prompt = (
            f"Missing fields: {missing}\n"
            f"Current profile: {json.dumps(current_profile_dict, ensure_ascii=False)}"
        )
        followup = call_llm_json(system_prompt, user_prompt, temperature=0.4, max_tokens=200)
        field = followup.get("field")
        question = followup.get("question")

        if not question or not field:
            break

        answer = "N/A"

        extract_system = (
            "Extract the value for the given field from the athlete's free-text answer.\n"
            "Use correct JSON typing (number, string, or array of strings as appropriate).\n"
            "Respond ONLY with JSON: {\"field\": \"<field_name>\", \"value\": <extracted value>}"
        )
        extract_user = f"Field: {field}\nAnswer: {answer}"
        extracted = call_llm_json(extract_system, extract_user, temperature=0.0, max_tokens=150)

        target_field = extracted.get("field", field)
        current_profile_dict[target_field] = extracted.get("value")

        missing = [m for m in missing if m != field]
        rounds += 1

    safe_data = {k: v for k, v in current_profile_dict.items() if k in PlayerProfile.model_fields}
    try:
        profile = PlayerProfile(**safe_data)
    except Exception as exc:
        logger.warning("Missing-information resolution validation issue: %s", exc)
        return profile

    return profile

def _run_scout_agent(agent_name, focus_description, profile, conversation):
    transcript = "\n".join(f"Q: {t['question']}\nA: {t['answer']}" for t in conversation)

    system_prompt = (
        f"You are the {agent_name} inside an AI sports scouting system.\n"
        f"Evaluate ONLY the following aspects: {focus_description}.\n"
        "Do NOT evaluate anything outside this scope.\n"
        "Base your evaluation strictly on the player profile and interview transcript "
        "provided; do not invent facts.\n"
        "Respond ONLY with a JSON object of the exact form:\n"
        "{\"score\": <float between 0 and 100>, \"summary\": \"<2-3 sentence summary>\", "
        "\"details\": {<relevant sub-scores or notes as key/value pairs>}, "
        "\"evidence\": [\"<short paraphrase drawn from the profile/transcript>\", ...]}"
    )
    user_prompt = (
        f"Player Profile:\n{profile.model_dump_json()}\n\n"
        f"Interview Transcript:\n{transcript}"
    )

    result = call_llm_json(system_prompt, user_prompt, temperature=0.3, max_tokens=1200)
    return AgentReport(
        agent_name=agent_name,
        score=clamp(result.get("score", 0.0), 0.0, 100.0),
        summary=str(result.get("summary", "")),
        details=result.get("details", {}) or {},
        evidence=result.get("evidence", []) or [],
    )

def run_experience_scout(profile, conversation):
    return _run_scout_agent(
        "Experience Scout Agent",
        "experience, competitions, club history, training consistency, competitive level",
        profile, conversation,
    )

def run_mentality_scout(profile, conversation):
    return _run_scout_agent(
        "Mentality Scout Agent",
        "discipline, motivation, leadership, learning ability, coachability, "
        "communication, handling pressure",
        profile, conversation,
    )

def run_achievement_scout(profile, conversation):
    return _run_scout_agent(
        "Achievement Scout Agent",
        "awards, tournaments, certificates, recognitions, milestones",
        profile, conversation,
    )

def run_development_scout(profile, conversation):
    return _run_scout_agent(
        "Development Scout Agent",
        "future potential, growth rate, development priorities, suitable competition "
        "level, long-term outlook",
        profile, conversation,
    )

def run_head_scout(profile, experience, mentality, achievement, development):
    system_prompt = (
        "You are the Head Scout Agent, the senior decision-maker of the scouting system.\n"
        "Merge the four specialist reports (experience, mentality, achievement, "
        "development) into one final evaluation. Weigh them holistically; do not simply "
        "average the scores without judgment.\n"
        "Respond ONLY with a JSON object of the exact form:\n"
        "{\"overall_score\": <float 0-100>, \"strengths\": [<string>, ...], "
        "\"weaknesses\": [<string>, ...], \"summary\": \"<paragraph>\", "
        "\"recommendation\": \"<clear actionable recommendation>\", "
        "\"development_plan\": [<string>, ...]}"
    )
    user_prompt = (
        f"Player: {profile.name}, Sport: {profile.sport}, Position: {profile.position}\n\n"
        f"Experience Report: {experience.model_dump_json()}\n\n"
        f"Mentality Report: {mentality.model_dump_json()}\n\n"
        f"Achievement Report: {achievement.model_dump_json()}\n\n"
        f"Development Report: {development.model_dump_json()}"
    )

    result = call_llm_json(system_prompt, user_prompt, temperature=0.3, max_tokens=1500)
    result["overall_score"] = clamp(result.get("overall_score", 0.0), 0.0, 100.0)
    return HeadScoutReport(**result)

def run_explainability_agent(experience, mentality, achievement, development, head_scout):
    system_prompt = (
        "You are the Explainability Agent.\n"
        "Explain clearly WHY each score was assigned, referencing the evidence collected "
        "during the interview. You must NEVER change or contradict any score, only explain it.\n"
        "Respond ONLY with JSON of the exact form:\n"
        "{\"explanations\": {\"experience\": \"...\", \"mentality\": \"...\", "
        "\"achievement\": \"...\", \"development\": \"...\", \"overall\": \"...\"}}"
    )
    user_prompt = (
        f"Experience: score={experience.score}, evidence={experience.evidence}\n"
        f"Mentality: score={mentality.score}, evidence={mentality.evidence}\n"
        f"Achievement: score={achievement.score}, evidence={achievement.evidence}\n"
        f"Development: score={development.score}, evidence={development.evidence}\n"
        f"Overall score: {head_scout.overall_score}, summary: {head_scout.summary}"
    )

    result = call_llm_json(system_prompt, user_prompt, temperature=0.3, max_tokens=1200)
    return ExplainabilityReport(**result)

def generate_pdf_report(profile, experience, mentality, achievement, development, head_scout, explainability, output_path=None):
    if output_path is None:
        safe_name = sanitize_filename(profile.name)
        output_path = os.path.join(OUTPUT_DIR, f"{safe_name}_scouting_report.pdf")

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
        leftMargin=1.5 * cm, rightMargin=1.5 * cm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"], fontSize=22,
        textColor=colors.HexColor("#0B2545"), alignment=TA_CENTER,
    )
    heading_style = ParagraphStyle(
        "HeadingStyle", parent=styles["Heading2"],
        textColor=colors.HexColor("#134074"), spaceBefore=14, spaceAfter=6,
    )
    body_style = ParagraphStyle("BodyStyle", parent=styles["BodyText"], fontSize=10.5, leading=15)
    small_style = ParagraphStyle("SmallStyle", parent=styles["BodyText"], fontSize=9, textColor=colors.grey)

    def _na(value):
        return str(value) if value not in (None, "", []) else "N/A"

    story = []
    story.append(Paragraph("SPOTME AI SCOUT", title_style))
    story.append(Paragraph("Professional Scouting Report", styles["Heading3"]))
    story.append(Spacer(1, 0.5 * cm))

    info_table_data = [
        ["Name", _na(profile.name), "Sport", _na(profile.sport)],
        ["Age", _na(profile.age), "Country", _na(profile.country)],
        ["Position", _na(profile.position), "Dominant Hand/Foot", _na(profile.dominant_hand_or_foot)],
        ["Height (cm)", _na(profile.height_cm), "Weight (kg)", _na(profile.weight_kg)],
        ["Current Team", _na(profile.current_team), "Years of Experience", _na(profile.years_of_experience)],
    ]
    info_table = Table(info_table_data, colWidths=[3.5 * cm, 5 * cm, 4.2 * cm, 4 * cm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#134074")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#134074")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
        ("TEXTCOLOR", (2, 0), (2, -1), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.6 * cm))

    story.append(Paragraph(f"Overall Score: {head_scout.overall_score:.1f} / 100", heading_style))
    story.append(Spacer(1, 0.2 * cm))

    scores_data = [
        ["Agent", "Score / 100"],
        ["Experience Scout", f"{experience.score:.1f}"],
        ["Mentality Scout", f"{mentality.score:.1f}"],
        ["Achievement Scout", f"{achievement.score:.1f}"],
        ["Development Scout", f"{development.score:.1f}"],
    ]
    scores_table = Table(scores_data, colWidths=[8 * cm, 4 * cm])
    scores_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B2545")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(scores_table)
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("Summary", heading_style))
    story.append(Paragraph(head_scout.summary or "N/A", body_style))

    story.append(Paragraph("Strengths", heading_style))
    for item in (head_scout.strengths or ["N/A"]):
        story.append(Paragraph(f"\u2022 {item}", body_style))

    story.append(Paragraph("Weaknesses", heading_style))
    for item in (head_scout.weaknesses or ["N/A"]):
        story.append(Paragraph(f"\u2022 {item}", body_style))

    story.append(Paragraph("Development Plan", heading_style))
    for item in (head_scout.development_plan or ["N/A"]):
        story.append(Paragraph(f"\u2022 {item}", body_style))

    story.append(Paragraph("Final Recommendation", heading_style))
    story.append(Paragraph(head_scout.recommendation or "N/A", body_style))

    story.append(PageBreak())
    story.append(Paragraph("AI Explanation", title_style))
    story.append(Spacer(1, 0.4 * cm))
    for key, explanation in explainability.explanations.items():
        story.append(Paragraph(key.replace("_", " ").capitalize(), heading_style))
        story.append(Paragraph(explanation, body_style))

    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by SPOTME AI Scout",
        small_style,
    ))

    doc.build(story)
    logger.info("PDF report saved to: %s", output_path)
    return output_path

def save_profile_and_reports_json(profile, experience, mentality, achievement, development, head_scout, explainability, output_path=None):
    if output_path is None:
        safe_name = sanitize_filename(profile.name)
        output_path = os.path.join(OUTPUT_DIR, f"{safe_name}_scouting_data.json")

    payload = {
        "profile": profile.model_dump(),
        "experience_report": experience.model_dump(),
        "mentality_report": mentality.model_dump(),
        "achievement_report": achievement.model_dump(),
        "development_report": development.model_dump(),
        "head_scout_report": head_scout.model_dump(),
        "explainability_report": explainability.model_dump(),
        "generated_at": datetime.now().isoformat(),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("JSON export saved to: %s", output_path)
    return output_path

def run_spotme_ai_scout_pipeline(sport):
    if sport not in SUPPORTED_SPORTS:
        raise ValueError(f"Unsupported sport '{sport}'. Choose from {SUPPORTED_SPORTS}")

    conversation = run_smart_interview(sport)
    if not conversation:
        raise RuntimeError("Interview produced no data; cannot build a profile.")

    logger.info("Building player profile...")
    profile = build_player_profile(conversation, sport)

    logger.info("Verifying profile...")
    verification = verify_profile(profile, sport)

    if verification.missing_fields:
        profile = resolve_missing_information(profile, verification, sport)
    else:
        safe_data = {
            k: v for k, v in verification.corrected_profile.items()
            if k in PlayerProfile.model_fields
        }
        profile = PlayerProfile(**safe_data)

    logger.info("Running Experience Scout Agent...")
    experience = run_experience_scout(profile, conversation)

    logger.info("Running Mentality Scout Agent...")
    mentality = run_mentality_scout(profile, conversation)

    logger.info("Running Achievement Scout Agent...")
    achievement = run_achievement_scout(profile, conversation)

    logger.info("Running Development Scout Agent...")
    development = run_development_scout(profile, conversation)

    logger.info("Running Head Scout Agent...")
    head_scout = run_head_scout(profile, experience, mentality, achievement, development)

    logger.info("Running Explainability Agent...")
    explainability = run_explainability_agent(experience, mentality, achievement, development, head_scout)

    logger.info("Generating PDF report...")
    pdf_path = generate_pdf_report(
        profile, experience, mentality, achievement, development, head_scout, explainability
    )

    logger.info("Saving JSON data...")
    json_path = save_profile_and_reports_json(
        profile, experience, mentality, achievement, development, head_scout, explainability
    )

    return {
        "conversation": conversation,
        "profile": profile,
        "experience": experience,
        "mentality": mentality,
        "achievement": achievement,
        "development": development,
        "head_scout": head_scout,
        "explainability": explainability,
        "pdf_path": pdf_path,
        "json_path": json_path,
    }

app = FastAPI(title="SPOTME AI Scout API", version="1.0.0")

class ScoutRequest(BaseModel):
    sport: str
    player_name: Optional[str] = None

class ScoutResponse(BaseModel):
    status: str
    player_name: Optional[str]
    sport: str
    overall_score: float
    summary: str
    recommendation: str
    pdf_path: str
    json_path: str
    reports: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    model: str
    supported_sports: List[str]

@app.get("/", response_model=Dict[str, str])
async def root():
    return {"message": "SPOTME AI Scout API is live!"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "healthy",
        "model": GROQ_MODEL,
        "supported_sports": SUPPORTED_SPORTS
    }

@app.post("/scout", response_model=ScoutResponse)
async def scout_player(request: ScoutRequest):
    if request.sport not in SUPPORTED_SPORTS:
        raise HTTPException(
            status_code=400,
            detail=f"Sport not supported. Choose from: {SUPPORTED_SPORTS}"
        )
    
    try:
        results = run_spotme_ai_scout_pipeline(request.sport)
        
        return {
            "status": "success",
            "player_name": results["profile"].name or "Unknown",
            "sport": request.sport,
            "overall_score": results["head_scout"].overall_score,
            "summary": results["head_scout"].summary,
            "recommendation": results["head_scout"].recommendation,
            "pdf_path": results["pdf_path"],
            "json_path": results["json_path"],
            "reports": {
                "experience": results["experience"].model_dump(),
                "mentality": results["mentality"].model_dump(),
                "achievement": results["achievement"].model_dump(),
                "development": results["development"].model_dump(),
                "head_scout": results["head_scout"].model_dump(),
                "explainability": results["explainability"].model_dump()
            }
        }
    except Exception as e:
        logger.error(f"Scouting error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reports")
async def list_reports():
    try:
        files = os.listdir(OUTPUT_DIR)
        reports = []
        for f in files:
            stat = os.stat(os.path.join(OUTPUT_DIR, f))
            reports.append({
                "name": f,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        return {"reports": reports}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/report/{file_name}")
async def get_report(file_name: str):
    file_path = os.path.join(OUTPUT_DIR, file_name)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    if file_name.endswith(".pdf"):
        media_type = "application/pdf"
    elif file_name.endswith(".json"):
        media_type = "application/json"
    else:
        media_type = "application/octet-stream"
    
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=file_name
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)