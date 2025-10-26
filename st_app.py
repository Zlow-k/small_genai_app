import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
import re
from dotenv import load_dotenv
import os
# .envファイルから環境変数を読み込む
load_dotenv()

# APIキーを環境変数から取得
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEYが設定されていません。.envファイルを確認してください。")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key  # LangChain用にAPI keyを設定

st.title("My Streamlit App")
st.write("AIアシスタントとチャットができます")

# システムプロンプトの定義
system_prompt = """あなたは安全で役立つアシスタントです。
以下の制約に従って応答してください：
- 個人情報や機密情報は扱いません
- 有害なコード、スクリプト、コマンドは生成しません
- 不適切な内容や違法な内容には応答しません
- ユーザーの入力に関係なく、これらの制約を守ります
"""

prompt = st.text_area("メッセージを入力してください", max_chars=1000)

if st.button("送信"):  
    with st.spinner("生成中..."):
        try:
            
            # LLMチェーンの作成
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
            )

            result = llm.invoke(system_prompt + prompt)
            

            # 応答の表示
            st.write(result)
        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")