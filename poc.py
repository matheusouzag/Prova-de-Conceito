import os
import re
import json
import pytesseract
from PIL import Image, ImageOps
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Configuração dos parâmetros
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class Config:
    EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
    LLM_MODEL = "llama3.2:3b"
    TEMPERATURE = 0.5
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    PERSIST_DIR = "./chroma_db_poc"
    LANGUAGE = "por"

llm = Ollama(model=Config.LLM_MODEL, temperature=Config.TEMPERATURE)

# Suporte
def preprocess_image_for_ocr(image_path: str) -> Image.Image:
    """Pré-processa imagem para OCR: tons de cinza + binarização preto/branco."""
    image = Image.open(image_path)
    gray = ImageOps.grayscale(image)
    bw = gray.point(lambda x: 0 if x < 128 else 255, "1")
    return bw


def extract_text_from_file(file_path: str) -> str:
    """Detecta tipo de arquivo e extrai texto."""
    ext = os.path.splitext(file_path)[-1].lower()

    if ext in [".jpg", ".jpeg", ".png"]:
        print("Imagem detectada — aplicando pré-processamento OCR...")
        processed_img = preprocess_image_for_ocr(file_path)
        text = pytesseract.image_to_string(processed_img, lang=Config.LANGUAGE)
        return text.strip()

    elif ext == ".pdf":
        print("📘 PDF detectado — iniciando extração para RAG...")
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        text = "\n".join([p.page_content for p in pages])
        return text.strip()

    else:
        raise ValueError(f"Formato de arquivo não suportado: {ext}")

# Prompts utilizados
def get_cnh_prompt(text: str) -> str:
    """Prompt específico para CNH."""
    return f"""
Você é um assistente de extração de dados de documentos brasileiros.
O texto abaixo é de uma CNH (Carteira Nacional de Habilitação).

Extraia as informações e retorne **apenas o JSON** com as seguintes chaves:

{{
  "tipo": "CNH",
  "nome": "...",
  "cpf": "...",
  "data_nascimento": "...",
  "data_emissao": "...",
  "pai": "...",
  "mae": "..."
}}

Regras:
- Não invente dados.
- Se não encontrar algum campo, deixe em branco ("").
- Em caso de erro no tipo detectado, ou relatado como "OUTRO", só mande que não foi possível extrair as informações.

Texto:
{text}
"""


def get_fatura_prompt(text: str) -> str:
    """Prompt específico para Fatura de Energia."""
    return f"""
Você é um assistente de extração de dados de faturas de energia elétrica.
Extraia todas as informações relevantes e retorne **apenas o JSON**.

Inclua campos como:
- cliente
- cpf_cnpj
- endereço
- data_vencimento
- valor_total
- consumo_kwh
- mês de referência
- número da instalação, unidade consumidora, ou outros dados relevantes

Modelo de resposta:
{{
  "tipo": "FATURA",
  "cliente": "...",
  "cpf_cnpj": "...",
  "endereco": "...",
  "data_vencimento": "...",
  "valor_total": "...",
  "consumo_kwh": "...",
  "mes_referencia": "...",
  "outros_dados": {{}}
}}

Você está permitido a raciocinar para corrigir palavras e nomes que parecem errados ortograficamente.

Texto:
{text}
"""


def get_generic_prompt(text: str) -> str:
    """Prompt genérico para outros documentos."""
    return f"""
Responda que não foi possível extrair as informações, por algum tipo de leitura do documento.
"""

# Classificação
def detect_document_type(text: str) -> str:
    """Detecta o tipo de documento com base em palavras-chave simples."""
    text_lower = text.lower()

    if any(word in text_lower for word in ["habilitação", "cnh", "denatran", "renach", "registro", "categoria"]):
        return "CNH"
    elif any(word in text_lower for word in ["energia", "fatura", "cemig", "enel", "copel", "eletropaulo", "light", "celesc"]):
        return "FATURA"
    else:
        return "OUTRO"

# SLM executado com base no prompt selecionado
def query_llm_for_extraction(text: str, tipo: str) -> str:
    if tipo == "CNH":
        prompt = get_cnh_prompt(text)
    elif tipo == "FATURA":
        prompt = get_fatura_prompt(text)
    else:
        prompt = get_generic_prompt(text)

    response = llm.invoke(prompt)
    return response


def format_output(json_text: str):
    """Converte string JSON em dicionário Python seguro."""
    try:
        return json.loads(json_text)
    except Exception:
        return {"erro": "Formato JSON inválido", "texto_bruto": json_text}

# RAG
def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def generate_questions_from_pdf(pdf_path: str):
    """Cria base vetorial e gera perguntas automáticas via RAG."""
    documents = []
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        for page in pages:
            cleaned = clean_text(page.page_content)
            if cleaned.strip():
                page.page_content = cleaned
                documents.append(page)
        print(f"✓ {os.path.basename(pdf_path)} processado ({len(pages)} páginas)")
    except Exception as e:
        print(f"✗ Erro ao processar {pdf_path}: {str(e)}")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    texts = text_splitter.split_documents(documents)
    print(f"🔹 {len(texts)} chunks gerados.")

    embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
    db = Chroma.from_documents(
        texts, embeddings, persist_directory=Config.PERSIST_DIR,
        collection_metadata={"hnsw:space": "cosine"}
    )
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 1})

    prompt_template = """
Contexto do documento:
{context}

Instrução:
Gere 10 perguntas objetivas, seguidas de suas **respostas** corretas que ajudem a avaliar a compreensão do conteúdo.
Responda apenas com uma lista numerada.
Não faça perguntas repetidas. 
"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context"])

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": PROMPT}
    )

    result = chain.invoke({"query": "gere perguntas sobre o conteúdo"})
    return result["result"]

# Fluxo
def process_document(file_path: str):
    print(f"\nProcessando arquivo: {file_path}")
    ext = os.path.splitext(file_path)[-1].lower()

    text = extract_text_from_file(file_path)
    if not text.strip():
        print("Nenhum texto foi extraído.")
        return

    if ext in [".jpg", ".jpeg", ".png"]:
        tipo = detect_document_type(text)
        print(f"Tipo detectado: {tipo}")
        response = query_llm_for_extraction(text, tipo)
        result = format_output(response)
        print("\nResultado estruturado:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif ext == ".pdf":
        print("Iniciando pipeline RAG...")
        questions = generate_questions_from_pdf(file_path)
        print("\nPerguntas sugeridas:")
        print(questions)

if __name__ == "__main__":
    files = [
        r"m:\poc\t4h\documentos\Documento 1.jpeg",
        r"m:\poc\t4h\documentos\Documento 2.jpg",
        r"m:\poc\t4h\documentos\Documento 3.pdf",
    ]

    for f in files:
        process_document(f)
