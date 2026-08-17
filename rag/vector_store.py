"""
Vector Store Module
Manages Qdrant collections and vector similarity search operations.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)


class QdrantStore:
    """
    Manages Qdrant vector database operations.
    Handles collection creation, document ingestion, and semantic search.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "terminal",
        embedding_dim: int = 3072,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim

        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(host=host, port=port)

        logger.info(f"Connected to Qdrant at {url or f'{host}:{port}'}")

    def ensure_collection(self) -> bool:
        """
        Create the Qdrant collection if it doesn't exist.
        Returns True if created, False if already existed.
        """
        try:
            self.client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' already exists")
            return False
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=self.embedding_dim,
                    distance=qdrant_models.Distance.COSINE,
                ),
                optimizers_config=qdrant_models.OptimizersConfigDiff(
                    indexing_threshold=0,
                ),
            )
            logger.info(f"Created collection '{self.collection_name}'")
            return True

    def add_documents(
        self,
        documents: List[Document],
        embeddings,
        batch_size: int = 50,
    ) -> List[str]:
        """
        Add documents with pre-computed embeddings to Qdrant.
        Returns list of inserted point IDs.
        """
        import time
        self.ensure_collection()

        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=embeddings,
        )

        all_ids = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            try:
                ids = vector_store.add_documents(batch)
                all_ids.extend(ids)
                logger.info(f"Ingested batch {i // batch_size + 1} ({len(batch)} chunks)")
            except Exception as e:
                logger.error(f"Failed to ingest batch {i // batch_size + 1}: {e}")
                print(f"\n  Error ingesting batch: {e}")
                break

        return all_ids

    def get_vector_store(self, embeddings) -> QdrantVectorStore:
        """Return a LangChain-compatible QdrantVectorStore instance."""
        self.ensure_collection()
        return QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=embeddings,
        )

    def delete_collection(self) -> bool:
        """Delete the current collection (reset all stored documents)."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """Return info about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "status": str(info.status),
            }
        except Exception as e:
            logger.warning(f"Could not fetch collection info: {e}")
            return {"points_count": 0}

    def document_exists(self, file_hash: str) -> bool:
        """Check if a document (by file hash) has already been ingested."""
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="metadata.file_hash",
                            match=qdrant_models.MatchValue(value=file_hash),
                        )
                    ]
                ),
                limit=1,
            )
            return len(results[0]) > 0
        except Exception:
            return False

    def list_ingested_documents(self) -> List[str]:
        """Return a list of unique filenames in the collection."""
        try:
            unique_filenames = set()
            offset = None

            while True:
                results, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in results:
                    payload = point.payload or {}
                    meta = payload.get("metadata", {})
                    fname = meta.get("source_filename") or meta.get("source", "")
                    if fname:
                        unique_filenames.add(fname)
                if next_offset is None:
                    break
                offset = next_offset

            return sorted(unique_filenames)
        except UnexpectedResponse as e:
            if getattr(e, "status_code", 404) == 404 or "Not found" in str(e):
                return []
            logger.warning(f"Could not list documents: {e}")
            return []
        except Exception as e:
            logger.warning(f"Could not list documents: {e}")
            return []
