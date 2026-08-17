import os
import time
import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

st.set_page_config(
    page_title="AI Technical Interviewer",
    page_icon="🎙️",
    layout="centered"
)

CANDIDATE_MODELS = [
    "gemini-flash-latest",
    "gemini-2.5-flash",
    "gemini-3.5-flash",
    "gemini-flash-lite-latest"
]

def extract_jd_text(pdf_file) -> str:
    try:
        reader = PdfReader(pdf_file)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        return text.strip()
    except Exception as e:
        st.sidebar.error(f"PDF error: {e}")
        return ""

def initialize_chat_with_fallback(client, system_prompt, init_message):
    last_error = None
    for model_name in CANDIDATE_MODELS:
        for attempt in range(1, 3):
            try:
                chat = client.chats.create(
                    model=model_name,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.7,
                    ),
                )
                response = chat.send_message(init_message)
                return chat, response.text, model_name
            except Exception as e:
                last_error = e
                err_str = str(e)
                if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str:
                    time.sleep(1.5 * attempt)
                    continue
                break
    raise last_error

def generate_report_with_fallback(chat_session, client, prompt, active_model):
    for attempt in range(1, 3):
        try:
            response = chat_session.send_message(prompt)
            return response.text
        except Exception as e:
            err_str = str(e)
            if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str:
                time.sleep(2 * attempt)
                continue
            break

    full_history = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages])
    composite_prompt = f"Transcript:\n{full_history}\n\nTask:\n{prompt}"

    for model_name in CANDIDATE_MODELS:
        if model_name == active_model:
            continue
        try:
            fallback_resp = client.models.generate_content(
                model=model_name,
                contents=composite_prompt
            )
            return fallback_resp.text
        except Exception:
            continue

    raise RuntimeError("All models busy.")

st.title("🎙️ AI Technical Interviewer")
st.caption("Powered by Google Gemini")

with st.sidebar:
    st.header("Settings")

    api_key_input = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="Enter your Gemini API key..."
    )
    api_key = api_key_input.strip() if api_key_input else ""
    st.markdown("[Get free API Key](https://aistudio.google.com/app/apikey)")

    target_role = st.text_input("Target Role", value="Junior Software Engineer")
    uploaded_pdf = st.file_uploader("Upload JD (PDF)", type=["pdf"])

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Reset", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    with col2:
        finish_interview = st.button("Finish", use_container_width=True)

if not api_key:
    st.info("👈 Enter your Gemini API Key in the sidebar to begin.")
    st.stop()

if "client" not in st.session_state or st.session_state.get("current_key") != api_key:
    st.session_state.client = genai.Client(api_key=api_key)
    st.session_state.current_key = api_key
    st.session_state.messages = []
    st.session_state.interview_report = None
    if "chat_initialized" in st.session_state:
        del st.session_state["chat_initialized"]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "interview_report" not in st.session_state:
    st.session_state.interview_report = None

if "active_model" not in st.session_state:
    st.session_state.active_model = CANDIDATE_MODELS[0]

if "audio_key" not in st.session_state:
    st.session_state.audio_key = 0

if "chat_initialized" not in st.session_state:
    try:
        jd_content = extract_jd_text(uploaded_pdf) if uploaded_pdf else ""
        if jd_content:
            st.sidebar.success("JD loaded")

        system_prompt = f"""
Role: Technical Interviewer for [{target_role}].
Context: {jd_content if jd_content else "Evaluate software engineering and API debugging fundamentals."}
Rules:
1. English only.
2. Under 100 words per response.
3. Ask ONE follow-up question per turn.
4. Use STAR framework.
"""

        with st.spinner("Connecting..."):
            chat, first_q, used_model = initialize_chat_with_fallback(
                client=st.session_state.client,
                system_prompt=system_prompt,
                init_message="Hello! Briefly introduce yourself and ask the first technical question."
            )
            st.session_state.chat = chat
            st.session_state.active_model = used_model
            st.session_state.messages.append({"role": "assistant", "content": first_q})
            st.session_state.chat_initialized = True

    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

if finish_interview:
    if len(st.session_state.messages) <= 1:
        st.sidebar.warning("Answer at least one question first.")
    else:
        with st.spinner("Generating report..."):
            try:
                report_prompt = """
Generate an evaluation report in Traditional Chinese (繁體中文):
1. Scorecard (STAR /10, Tech Depth /10, Communication /10)
2. Strengths
3. Areas for Improvement
4. Exemplary STAR Model Answer
"""
                report_text = generate_report_with_fallback(
                    chat_session=st.session_state.chat,
                    client=st.session_state.client,
                    prompt=report_prompt,
                    active_model=st.session_state.active_model
                )
                st.session_state.interview_report = report_text
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

if st.session_state.interview_report:
    st.success("Interview Complete:")
    st.markdown(st.session_state.interview_report)
    st.download_button(
        label="Download Report",
        data=st.session_state.interview_report,
        file_name="report.md",
        mime="text/markdown"
    )
    st.divider()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

st.divider()
user_audio = st.audio_input("Record voice", key=f"audio_recorder_{st.session_state.audio_key}")
user_text = st.chat_input("Type answer...")

user_payload = None
if user_text:
    user_payload = user_text
elif user_audio:
    audio_bytes = user_audio.read()
    if audio_bytes:
        user_payload = types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")

if user_payload:
    if isinstance(user_payload, str):
        st.session_state.messages.append({"role": "user", "content": user_payload})
    else:
        st.session_state.messages.append({"role": "user", "content": "🎙️ [Voice response]"})

    try:
        with st.chat_message("assistant"):
            with st.spinner("Evaluating..."):
                response = st.session_state.chat.send_message(user_payload)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
                if not isinstance(user_payload, str):
                    st.session_state.audio_key += 1
                st.rerun()
    except Exception as e:
        st.error(f"Error: {e}")