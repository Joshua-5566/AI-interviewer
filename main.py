import os
import torch

# 1. 自動授權信任 Silero VAD 儲存庫，避免終端機彈出 y/N 詢問
torch.hub.set_dir(torch.hub.get_dir())
try:
    torch.hub.trust_repo("snakers4/silero-vad")
except Exception:
    pass

import pyttsx3
from google import genai
from google.genai import types
from RealtimeSTT import AudioToTextRecorder


def main():
    # 2. 初始化 Gemini Client
    # 建議在 PowerShell 先設定：$env:GEMINI_API_KEY="your_api_key"
    # 如果要直接寫在程式碼中，請填入 your_api_key
    client = genai.Client()

    SYSTEM_PROMPT = (
        "You are a technical interviewer for an entry-level software engineer role. "
        "Evaluate candidate answers briefly using the STAR method, give actionable feedback, "
        "and ask one relevant follow-up question at a time. Keep responses concise and conversational."
    )

    chat = client.chats.create(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.7,
        ),
    )

    # 3. 語音合成引擎 (TTS)
    engine = pyttsx3.init()

    def speak(text: str):
        print(f"\nAI: {text}")
        engine.say(text)
        engine.runAndWait()

    # 4. 初始化語音辨識 (STT) - 調整參數讓 AI 允許更長的說話停頓
    print("Loading Speech Recognition Model... (This takes a few seconds on first launch)")
    recorder = AudioToTextRecorder(
        model="tiny.en",
        post_speech_silence_duration=1.2,  # 說完話後需靜音 1.2 秒才認定回答完畢
        min_length_of_recording=1.0,       # 避免吸氣或背景音誤觸發
        silero_sensitivity=0.4             # VAD 靈敏度 (0~1)
    )

    initial_question = (
        "Hello! I am your AI interviewer. Let's begin: "
        "Tell me about a challenging technical bug you solved recently."
    )
    speak(initial_question)

    # 5. 主對話迴圈
    while True:
        print("\n[Listening... Speak into your microphone]")
        user_text = recorder.text().strip()

        if not user_text:
            continue

        print(f"You: {user_text}")

        # 檢查退出指令
        if user_text.lower() in ["exit", "quit", "stop"]:
            speak("Thank you for your time. The interview is now complete.")
            break

        # 將回答傳送給 Gemini 進行評估並取得追問
        response = chat.send_message(user_text)
        speak(response.text)


# 防護 Windows multiprocessing 的關鍵語句
if __name__ == "__main__":
    main()