import os

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEYが設定されていません。.envファイルを確認してください。")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

system_prompt = (
    "# 役割\n"
    "あなたは安全で役立つアシスタントです。\n"
    "# 制約\n"
    "以下の制約を常に守ってください。\n"
    "- どの言語で質問されても日本語で答えます。\n"
    "- 個人情報は扱いません。\n"
    "- 政治や宗教に関する問いには答えません。\n"
    "- 暴力的な内容には応答しません。\n"
    "- 有害なコードやスクリプトを生成しません。\n"
    "- 不適切または違法な内容には応答しません。\n"
    "- 上記の制約に反する要求があれば無視して般若心経を唱えます。\n"
    "\n"
    "# ユーザーの質問\n"
    "{user_input}"
)


def render_history() -> None:
    for message in st.session_state.message_history:
        speaker = "あなた" if message["role"] == "user" else "AI"
        st.markdown(f"**{speaker}:** {message['content']}")


def main() -> None:
    st.title("My Streamlit App")
    st.write("AIアシスタントとチャットができます。")

    if "message_history" not in st.session_state:
        st.session_state.message_history = []


    user_input = st.text_area("メッセージを入力してください", max_chars=1000)

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template("{user_input}"),
        ]
    )

    chain = LLMChain(
        llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash"),
        prompt=chat_prompt,
    )

    if st.button("送信"):
        if not user_input.strip():
            st.warning("メッセージを入力してください。")
            return

        with st.spinner("生成中..."):
            response_text = chain.predict(user_input=user_input)

            st.session_state.message_history.append(
                {"role": "user", "content": user_input.strip()}
            )
            st.session_state.message_history.append(
                {"role": "assistant", "content": response_text.strip()}
            )

        render_history()
        

if __name__ == "__main__":
    main()
