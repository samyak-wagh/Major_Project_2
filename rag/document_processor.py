"""
Document Processor Module
Handles PDF loading, text extraction, and intelligent chunking.
"""

import os
import hashlib
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import fitz  # PyMuPDF
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Handles PDF document processing including loading, cleaning,
    and chunking for RAG pipeline ingestion.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            add_start_index=True,
        )

    def load_pdf_with_llamaparse(self, file_path: str) -> List[Document]:
        """Attempt to load PDF using LlamaParse if API key is present."""
        try:
            from llama_parse import LlamaParse
            parser = LlamaParse(
                api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
                result_type="markdown",  # Markdown format is best for tables/code
                verbose=True
            )
            parsed_docs = parser.load_data(file_path)
            
            # Convert LlamaIndex documents to LangChain documents
            documents = []
            for i, doc in enumerate(parsed_docs):
                documents.append(
                    Document(
                        page_content=doc.text,
                        metadata={
                            "source": file_path,
                            "page": i,
                        }
                    )
                )
            logger.info(f"Loaded document from {file_path} via LlamaParse")
            return documents
        except ImportError:
            logger.warning("llama-parse not installed. Falling back to PyMuPDF.")
            return []
        except Exception as e:
            logger.error(f"LlamaParse failed: {e}. Falling back to PyMuPDF.")
            return []

    def load_pdf(self, file_path: str) -> List[Document]:
        """Load a PDF file. Uses LlamaParse if available, otherwise PyMuPDF."""
        
        # 1. Try LlamaParse first (if API key is configured)
        if os.getenv("LLAMA_CLOUD_API_KEY"):
            logger.info("LLAMA_CLOUD_API_KEY found, attempting extraction via LlamaParse...")
            docs = self.load_pdf_with_llamaparse(file_path)
            if docs:
                return docs
        
        # 2. Fallback to PyMuPDF (fitz)
        try:
            doc = fitz.open(file_path)
            documents = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                # Clean text before creating the document
                text = self.clean_text(text)
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_path,
                            "page": page_num,
                        }
                    )
                )
            logger.info(f"Loaded {len(documents)} pages from {file_path} via PyMuPDF")
            return documents
        except Exception as e:
            logger.error(f"Error loading PDF {file_path} via PyMuPDF: {e}")
            # 3. Final Fallback to PyPDFLoader
            try:
                loader = PyPDFLoader(file_path)
                return loader.load()
            except Exception:
                raise e

    def load_pdf_from_path(self, file_path: str) -> List[Document]:
        """Load PDF from a filesystem path, attach metadata, and filter blank pages."""
        path = Path(file_path)
        file_bytes = path.read_bytes()
        documents = self.load_pdf(file_path)
        file_hash = self._compute_hash(file_bytes)

        non_empty = []
        for doc in documents:
            doc.metadata["source_filename"] = path.name
            doc.metadata["file_hash"] = file_hash
            # Keep only pages that have actual text content
            if doc.page_content and doc.page_content.strip():
                non_empty.append(doc)

        skipped = len(documents) - len(non_empty)
        if skipped:
            logger.warning(
                f"{skipped}/{len(documents)} pages had no extractable text "
                f"(possible image/scanned pages) in {path.name}"
            )
        return non_empty, len(documents)

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks for embedding."""
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} pages")

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_total"] = len(chunks)
            chunk.metadata["chunk_char_count"] = len(chunk.page_content)

        return chunks

    def process_pdf_path(self, file_path: str) -> Dict[str, Any]:
        """
        Full pipeline: load PDF from path → extract pages → chunk text.
        Returns a dict with chunks and metadata.
        """
        documents, raw_page_count = self.load_pdf_from_path(file_path)
        chunks = self.chunk_documents(documents)
        file_hash = self._compute_hash(Path(file_path).read_bytes())
        extracted_chars = sum(len(d.page_content) for d in documents)

        return {
            "filename": Path(file_path).name,
            "file_hash": file_hash,
            "total_pages": raw_page_count,          # original PDF page count
            "text_pages": len(documents),            # pages with actual text
            "total_chunks": len(chunks),
            "extracted_chars": extracted_chars,
            "chunks": chunks,
        }

    @staticmethod
    def _compute_hash(data: bytes) -> str:
        """Compute SHA-256 hash of file bytes for deduplication."""
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def clean_text(text: str) -> str:
        """Thorough text cleaning: remove nulls, normalize whitespace, keep readable chars."""
        import re
        if not text:
            return ""
        # Remove null bytes and non-printable chars
        text = text.replace("\x00", "")
        # Normalize all types of whitespace (tabs, newlines, etc) to single space
        text = re.sub(r"\s+", " ", text)
        return text.strip()
