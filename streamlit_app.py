import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from tempfile import NamedTemporaryFile
from langchain.schema import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
# 임베딩 모델
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# PDF 텍스트 추출 함수
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text_per_page = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            text_per_page.append({"page": i+1, "text": text})
    return text_per_page

# 요약 함수 (청크 기반)
def summarize_chunks(chunks):
    summaries = []
    for idx, chunk in enumerate(chunks):
        prompt = f"다음은 강의자료의 일부입니다. 간결하고 핵심적으로 요약해주세요 (Part {idx+1}):\n\n{chunk}"
        res = llm([HumanMessage(content=prompt)])
        summaries.append(res.content)
    return "\n\n".join(summaries)

# 연습문제 생성 함수
def generate_quiz(summary, num_questions=20):
    prompt = f"다음 강의자료 요약을 기반으로 객관식, 주관식, O/X, 단답형 문제를 골고루 섞어 총 {num_questions}개의 연습문제를 생성해줘.\n\n요약:\n{summary}"
    res = llm([HumanMessage(content=prompt)])
    return res.content

# Chroma에 저장
def save_to_chroma(split_docs):
    vectordb = Chroma.from_documents(split_docs, embedding=embedding_model, persist_directory="chroma_db")
    vectordb.persist()

# 검색
def search_chroma(query):
    vectordb = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)
    retriever = vectordb.as_retriever()
    docs = retriever.get_relevant_documents(query)
    return docs

# LLM 설정
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.7,
)

st.title("📚 강의자료 요약 및 연습문제 생성기 (멀티 PDF 지원)")

uploaded_files = st.file_uploader("📤 여러 PDF 업로드", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_docs = []
    for uploaded_file in uploaded_files:
        st.info(f"✅ {uploaded_file.name} 업로드 완료")
        text_per_page = extract_text_from_pdf(uploaded_file)
        for item in text_per_page:
            all_docs.append({"text": item["text"], "source": uploaded_file.name, "page": item["page"]})

    # 텍스트만 추출
    full_text = "\n\n".join([doc["text"] for doc in all_docs])

    # 텍스트 청크 분할
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    split_docs = splitter.create_documents([doc["text"] for doc in all_docs])

    with st.spinner("요약 중입니다..."):
        summary = summarize_chunks([doc.page_content for doc in split_docs])
        st.subheader("📝 요약 결과")
        st.write(summary)

    with st.spinner("연습문제 생성 중입니다..."):
        quiz = generate_quiz(summary, num_questions=20)
        st.subheader("❓ 연습문제")
        st.write(quiz)

    with st.spinner("ChromaDB 저장 중입니다..."):
        save_to_chroma(split_docs)
        st.success("✅ ChromaDB 저장 완료")

    query = st.text_input("🤖 질문을 입력하세요 (강의자료 관련)")

    if query:
        with st.spinner("답변 생성 중입니다..."):
            vectordb = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)
            retriever = vectordb.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
            response = qa_chain.run(query)
            st.subheader("💬 Gemini의 답변")
            st.write(response)

    keyword = st.text_input("🔍 특정 키워드가 어떤 문서 몇 페이지에 있는지 찾기")
    if keyword:
        with st.spinner("검색 중입니다..."):
            results = search_chroma(keyword)
            st.subheader("🔎 검색 결과")
            for i, doc in enumerate(results):
                st.write(f"{i+1}. 일부 내용: {doc.page_content[:300]}...")
