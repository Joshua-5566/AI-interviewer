import os
import time
import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

# 1. 頁面基本配置
st.set_page_config(
    page_title="AI 語音技術面試官",
    page_icon="🎙️",
    layout="centered"
)

# PDF 內文讀取函式
def extract_jd_text(pdf_file) -> str:
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"PDF 解析失敗: {e}")
        return ""

# 備援模型清單（依優先順序嘗試）
CANDIDATE_MODELS = [
    "gemini-flash-latest",
    "gemini-2.5-flash",
    "gemini-3.5-flash",
    "gemini-flash-lite-latest"
]

# 具備自動重試與模型備援的初始化函式
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

# 具備自動重試與模型備援的報告生成函式
def generate_report_with_fallback(chat_session, client, prompt, active_model):
    # 優先嘗試當前 session
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

    # 若當前 session 失敗，組合歷史紀錄並切換其他模型生成
    full_history = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages])
    composite_prompt = f"以下是完整的面試對話紀錄：\n{full_history}\n\n{prompt}"

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

    raise RuntimeError("所有可用模型目前皆遭遇尖峰流量，請稍候 10 秒後再次點擊。")

st.title("🎙️ AI 技術面試官")
st.caption("由 Gemini API 驅動的互動式技術面試系統")

# 2. 側邊欄設定
with st.sidebar:
    st.header("⚙️ 面試設定")

    default_key = ""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            default_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    if not default_key:
        default_key = os.environ.get("GEMINI_API_KEY", "")

    api_key = st.text_input("Gemini API Key", value=default_key, type="password")
    target_role = st.text_input("面試職位", value="初級軟體工程師 (Junior Software Engineer)")
    uploaded_pdf = st.file_uploader("上傳 Job Description (PDF)", type=["pdf"])

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 重置面試", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    with col2:
        finish_interview = st.button("📊 結束面試", use_container_width=True)

if not api_key:
    st.warning("⚠️ 請在左側輸入你的 Gemini API Key 以開始面試。")
    st.stop()

# 3. 初始化 Client 與 Session State
if "client" not in st.session_state:
    st.session_state.client = genai.Client(api_key=api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "interview_report" not in st.session_state:
    st.session_state.interview_report = None

if "active_model" not in st.session_state:
    st.session_state.active_model = CANDIDATE_MODELS[0]

if "chat_initialized" not in st.session_state:
    try:
        jd_content = ""
        if uploaded_pdf:
            jd_content = extract_jd_text(uploaded_pdf)
            st.sidebar.success("✅ 成功解析 Job Description PDF！")

        system_prompt = f"""
        你是一位針對【{target_role}】職位專業且嚴謹的技術面試官。

        === 職缺描述 (Job Description) ===
        {jd_content if jd_content else "針對一般軟體工程師基礎技術、系統設計與 STAR 原則進行考察。"}
        =================================

        規則：
        1. 請使用英文進行面試發問。
        2. 針對候選人的回答進行簡短評價（特別關注是否符合 STAR 原則：Situation, Task, Action, Result）。
        3. 每次只提出一個具深度且連貫的技術追問，語氣專業且自然。
        """

        with st.spinner("AI 面試官正在準備第一個提問（連線中）..."):
            chat, first_question, used_model = initialize_chat_with_fallback(
                client=st.session_state.client,
                system_prompt=system_prompt,
                init_message="Hello! Please introduce yourself briefly and ask the first technical interview question based on the role/JD."
            )
            st.session_state.chat = chat
            st.session_state.active_model = used_model
            st.session_state.messages.append({"role": "assistant", "content": first_question})
            st.session_state.chat_initialized = True

    except Exception as e:
        st.error(f"❌ 初始化 Gemini 對話失敗: {e}")
        st.info("💡 提示：Google 伺服器目前流量較大，請點擊左側「🔄 重置面試」重試。")
        st.stop()

# 4. 結束面試並生成評估報告
if finish_interview:
    if len(st.session_state.messages) <= 1:
        st.sidebar.warning("⚠️ 請先回答至少一個問題再結束面試。")
    else:
        with st.spinner("🔍 正在分析面試對話並生成 STAR 評估報告（若遇尖峰將自動重試）..."):
            try:
                report_prompt = """
                你現在是資深技術面試主管。請根據剛才的所有面試問答紀錄，輸出結構化的【面試評估報告】。

                請包含以下內容：
                1. 綜合評分（STAR 原則評分 /10、技術深度評分 /10、溝通表達評分 /10）
                2. 表現亮點（候選人回答得好的具體技術細節與情境架構）
                3. 改進建議（哪些問題缺乏具體數據、缺少 Action 或結果不明確）
                4. 推薦參考回答範例（挑選候選人回答最薄弱的一題，提供一段符合 STAR 原則的滿分示範）

                請使用繁體中文清晰輸出，排版需整齊易讀。
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
                st.error(f"❌ 生成報告失敗: {e}")

# 顯示評估報告
if st.session_state.interview_report:
    st.success("🎉 面試已結束！以下是你的專屬面試評估報告：")
    st.markdown(st.session_state.interview_report)
    st.download_button(
        label="📥 下載面試評估報告 (Markdown)",
        data=st.session_state.interview_report,
        file_name="interview_report.md",
        mime="text/markdown"
    )
    st.divider()

# 5. 渲染對話紀錄
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 6. 用戶輸入區
st.divider()
user_audio = st.audio_input("🎤 點擊進行語音回答（說完再點一次停止）")
user_text = st.chat_input("或在此輸入你的英文回答...")

user_payload = None

if user_text:
    user_payload = user_text
elif user_audio:
    audio_bytes = user_audio.read()
    user_payload = types.Part.from_bytes(
        data=audio_bytes,
        mime_type="audio/wav"
    )

if user_payload:
    if isinstance(user_payload, str):
        st.session_state.messages.append({"role": "user", "content": user_payload})
    else:
        st.session_state.messages.append({"role": "user", "content": "🎙️ [發送了語音回答]"})

    try:
        with st.chat_message("assistant"):
            with st.spinner("面試官思考中..."):
                response = st.session_state.chat.send_message(user_payload)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
                st.rerun()
    except Exception as e:
        st.error(f"發送訊息失敗: {e}")