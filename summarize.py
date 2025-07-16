from langchain_google_genai import ChatGoogleGenerativeAI

def summarize_text(text: str, api_key: str) -> str:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.7,
    )
    prompt = f"다음 내용을 간략히 요약해 주세요:\n{text}"
    response = llm(prompt)  # 여기서 response는 str
    return response  # 그냥 바로 반환
