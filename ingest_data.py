# ingest_data.py

import os
from PIL import Image
import pytesseract

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# config 파일에서 모델 이름과 경로를 가져옵니다.
import config

# Tesseract 실행 파일 경로 설정 (Windows 사용자의 경우 필요할 수 있음)
# 예: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_image(image_path: str) -> Document:
    """이미지 파일에서 텍스트를 추출하고 LangChain 문서로 반환합니다."""
    try:
        image = Image.open(image_path)
        # 한국어와 영어를 모두 인식하도록 설정
        text = pytesseract.image_to_string(image, lang='kor+eng')
        return Document(page_content=text, metadata={"source": os.path.basename(image_path)})
    except Exception as e:
        print(f"⚠️ 이미지 처리 오류 {image_path}: {e}")
        return None

def create_vector_store(
    data_path: str = "data/",
    vector_store_path: str = config.VECTOR_STORE_PATH,
    embedding_model: str = config.EMBEDDING_MODEL
):
    """
    지정된 경로의 문서(PDF, 이미지)를 로드하여 FAISS 벡터 저장소를 생성합니다.
    """
    if not os.path.exists(data_path):
        print(f"❌ 데이터 폴더 '{data_path}'를 찾을 수 없습니다. 폴더를 생성하고 파일을 넣어주세요.")
        return

    all_docs = []
    
    # 1. PDF 및 이미지 파일 로드
    print(f"📂 '{data_path}' 폴더에서 문서 로드를 시작합니다...")
    for filename in os.listdir(data_path):
        full_path = os.path.join(data_path, filename)
        if filename.lower().endswith(".pdf"):
            try:
                loader = PyMuPDFLoader(full_path)
                docs = loader.load()
                all_docs.extend(docs)
                print(f"  📄 PDF 로드 완료: {filename} ({len(docs)} 페이지)")
            except Exception as e:
                print(f"  ⚠️ PDF 로드 실패: {filename} ({e})")
        elif filename.lower().endswith((".jpg", ".png", ".jpeg")):
            doc = extract_text_from_image(full_path)
            if doc:
                all_docs.append(doc)
                print(f"  🖼️ 이미지 처리 완료: {filename}")

    if not all_docs:
        print("🚫 처리할 문서가 없습니다. 'data' 폴더를 확인해주세요.")
        return

    # 2. 문서 분할
    print(f"\n🌀 총 {len(all_docs)}개의 문서를 텍스트 청크로 분할합니다...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    print(f"  ✅ 총 {len(chunks)}개의 청크 생성 완료.")

    # 3. 임베딩 및 벡터 저장소 생성
    print(f"\n🧠 임베딩 모델 '{embedding_model}'을 사용하여 벡터화를 시작합니다...")
    embedding = HuggingFaceEmbeddings(model_name=embedding_model)
    
    db = FAISS.from_documents(chunks, embedding)
    db.save_local(vector_store_path)
    
    print(f"\n✨ FAISS 벡터 저장소 생성이 완료되었습니다!")
    print(f"  📍 저장 경로: '{vector_store_path}'")


if __name__ == "__main__":
    # 이 스크립트를 직접 실행하면 벡터 저장소를 생성합니다.
    create_vector_store()
