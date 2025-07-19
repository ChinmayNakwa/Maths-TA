import os
import fitz
from pdf2image import convert_from_path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from pathlib import Path
# Correct the import paths based on the new structure
from backend.config import settings
from ..database.astra_db_connection import get_vector_store
from .cloudinary_uploader import upload_image_to_cloudinary

def process_pdf_book():
    """
    Processes the PDF book specified in settings:
    1. Extracts text and page images.
    2. Chunks the text.
    3. Creates LangChain Document objects with rich metadata.
    4. Adds the documents to the AstraDB vector store.
    """

    pdf_path = Path(settings.PDF_PATH)
    if not pdf_path.exists():
        print(f"Error: PDF file not found at '{pdf_path}'")
        return 
    
    print(f"--- Starting PDF Book Ingestion: {pdf_path.name} ---")

    image_output_dir = Path(settings.IMAGE_OUTPUT_DIR)
    image_output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract content page by page and collect all documents
    all_page_docs = []
    doc_pymupdf = fitz.open(pdf_path)

    for page_num, page in enumerate(doc_pymupdf):
        page_text = page.get_text()
        print(f"  - Processing Page {page_num + 1}/{len(doc_pymupdf)}")

        if not page_text.strip():
            continue

        # 2. Render the page as an image
        image_filename_base = f"book_{pdf_path.stem}_page_{page_num + 1}"
        image_path = image_output_dir / f"{image_filename_base}.png"
        image_url = ""

        # This logic is correct
        if not image_path.exists():
            images = convert_from_path(str(pdf_path), first_page=page_num + 1, last_page=page_num + 1)
            if images: 
                images[0].save(image_path, "PNG")

        if image_path.exists():
            image_url = upload_image_to_cloudinary(image_path, public_id=image_filename_base)
            if image_url:
                print(f"    - Uploaded page {page_num + 1} image to Cloudinary.")

        # 3. Create a LangChain Document and add it to our list
        all_page_docs.append(Document(
            page_content=page_text,
            metadata = {
                "source": "PROBABILITY_FOR_COMPUTER_SCIENTIST",
                "file_name": pdf_path.name,
                "page_number": page_num + 1,
                "image_path": image_url
            }
        ))
    
    # --- STEPS 4 & 5 ARE MOVED OUTSIDE THE LOOP ---

    print(f"\nFinished processing all {len(all_page_docs)} pages.")

    # 4. Split ALL the documents at once
    print("Splitting documents into smaller chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = settings.CHUNK_SIZE,
        chunk_overlap = settings.CHUNK_OVERLAP,
    )
    chunked_docs = text_splitter.split_documents(all_page_docs)
    print(f"Split {len(all_page_docs)} pages into {len(chunked_docs)} chunks.")

    # 5. Add all chunks to the vector store in batches
    print("Initializing vector store and adding all documents...")
    vector_store = get_vector_store()

    batch_size = 20
    for i in range(0, len(chunked_docs), batch_size):
        batch = chunked_docs[i:i+batch_size]
        # It's good practice to wrap DB calls in a try/except
        try:
            vector_store.add_documents(batch)
            print(f"  - Added batch {i//batch_size + 1}/{(len(chunked_docs) + batch_size - 1)//batch_size}")
        except Exception as e:
            print(f"  - FAILED to add batch {i//batch_size + 1}: {e}")

    print("\n--- PDF Book Ingestion Complete! ---")

# The corrected __main__ block
if __name__ == '__main__':
    process_pdf_book()