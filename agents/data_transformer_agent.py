# 데이터를 변환하고 벡터 저장소로 저장하는 에이전트

import os
from PIL import Image
import pytesseract

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# 이미지 파일에서 텍스트를 추출하고 LangChain 문서로 반환하는 함수
def extract_text_from_image(image_path: str) -> Document:
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang='kor+eng')
    return Document(page_content=text, metadata={"source": image_path})

# 메인 함수: PDF와 이미지 파일을 로드하고 벡터 저장소로 변환
def ingest_documents(
    data_path: str = "data/",
    output_path: str = "vector_store/",
    model_name="jhgan/ko-sbert-nli"
):

    print(f"📂 '{data_path}' 내 PDF 로딩 중...")
    docs = []

    # 📄 PDF 처리
    pdf_files = [f for f in os.listdir(data_path) if f.endswith(".pdf")]
    if not pdf_files:
        print("🚫 PDF 파일이 존재하지 않습니다.")
        return {"message": "PDF 없음"}

    for filename in pdf_files:
        full_path = os.path.join(data_path, filename)
        try:
            loader = PyMuPDFLoader(full_path)
            loaded_docs = loader.load()
            docs.extend(loaded_docs)
            print(f"✅ {filename}: {len(loaded_docs)}개 문서 로드됨")
        except Exception as e:
            print(f"⚠️ {filename} 로드 실패: {e}")


    # 🖼️ 이미지 처리
    image_files = [f for f in os.listdir(data_path) if f.endswith((".jpg", ".png", ".jpeg"))]
    for filename in image_files:
        full_path = os.path.join(data_path, filename)
        try:
            doc = extract_text_from_image(full_path)
            docs.append(doc)
            print(f"✅ 이미지 {filename}: 텍스트 추출 완료")
        except Exception as e:
            print(f"⚠️ 이미지 {filename} OCR 실패: {e}")


    print(f"총 {len(docs)}개 문서 로드됨")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"총 {len(chunks)}개 청크 생성됨")

    print(f"임베딩 모델: {model_name}")
    embedding = HuggingFaceEmbeddings(model_name=model_name)

    print(f"FAISS 벡터 저장소 생성 중...")
    db = FAISS.from_documents(chunks, embedding)
    db.save_local(output_path)
    print(f"✅ 저장 완료: {output_path}")

    return {"message": f"{len(chunks)} chunks 저장 완료"}

# CLI 실행용
if __name__ == "__main__":
    ingest_documents()
