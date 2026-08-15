import os
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


st.title("🎙️ AI 技術面試官")
st.caption("由 Gemini API 驅動的互動式技術面試系統")

# 2. 側邊欄設定（包含結束面試按鈕）
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

    # 左右並排放置兩個按鈕
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 重置面試", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    with col2:
        finish_interview = st.button("📊 結束面試", use_container_width=True)

# 檢查是否有 API Key
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

        st.session_state.chat = st.session_state.client.chats.create(
            model="gemini-flash-latest",
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.7,
            ),
        )

        with st.spinner("AI 面試官正在準備第一個提問..."):
            init_response = st.session_state.chat.send_message(
                "Hello! Please introduce yourself briefly and ask the first technical interview question based on the role/JD."
            )
            st.session_state.messages.append({"role": "assistant", "content": init_response.text})
            st.session_state.chat_initialized = True

    except Exception as e:
        st.error(f"❌ 初始化 Gemini 對話失敗: {e}")
        st.stop()

# 4. 生成評估報告邏輯（點擊結束按鈕時觸發）
if finish_interview:
    if len(st.session_state.messages) <= 1:
        st.sidebar.warning("⚠️ 請先回答至少一個問題再結束面試！")
    else:
        with st.spinner("🔍 正在根據面試對話與 STAR 架構生成詳細評估報告..."):
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
                eval_response = st.session_state.chat.send_message(report_prompt)
                st.session_state.interview_report = eval_response.text
            except Exception as e:
                st.error(f"生成報告失敗: {e}")

# 若已生成報告，顯示於網頁頂部
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
    # 記錄使用者訊息
    if isinstance(user_payload, str):
        st.session_state.messages.append({"role": "user", "content": user_payload})
    else:
        st.session_state.messages.append({"role": "user", "content": "🎙️ [發送了語音回答]"})

    # 傳送給 Gemini 並取得回應
    try:
        with st.chat_message("assistant"):
            with st.spinner("面試官思考中..."):
                response = st.session_state.chat.send_message(user_payload)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
                st.rerun()
    except Exception as e:
        st.error(f"發送訊息失敗: {e}")