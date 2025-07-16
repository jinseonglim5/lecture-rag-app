from langchain_google_genai import ChatGoogleGenerativeAI

def generate_quiz(summary: str, api_key: str, quiz_type: str = "객관식", num_questions: int = 3) -> str:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.7,
    )
    prompt = f"아래 내용을 바탕으로 {quiz_type} 문제 {num_questions}개를 만들어 주세요.\n{summary}"
    response = llm(prompt)
    return response  # str로 반환됨
