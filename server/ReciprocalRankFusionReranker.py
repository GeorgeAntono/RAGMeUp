from typing import Sequence, List, Union
from langchain_core.documents import Document, BaseDocumentCompressor
from langchain_core.callbacks import Callbacks


class ReciprocalRankFusionReranker(BaseDocumentCompressor):
    """
    A reranker that implements Reciprocal Rank Fusion (RRF) to merge multiple ranked document lists.
    """

    k: int = 60
    """The parameter in the RRF formula to control the score."""
    top_n: int = 10
    """Number of top documents to return after reranking."""

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"

    def compress_documents(
            self,
            ranked_lists: List[List[Union[Document, tuple]]],  # Handle both Document and tuple inputs
            query: str = None,
            callbacks: Callbacks = None,
    ) -> Sequence[Document]:
        """
        Rerank documents using Reciprocal Rank Fusion (RRF).

        Args:
            ranked_lists: A list of ranked document lists from different retrievers.
            query: Optional query string (not used in this method but kept for compatibility).
            callbacks: Optional callbacks for tracing and debugging.

        Returns:
            A list of top_n documents reranked using RRF.
        """
        fused_scores = {}

        # Iterate over ranked lists
        for ranked_list in ranked_lists:
            for rank, item in enumerate(ranked_list):
                # Ensure we are working with a Document object
                doc = item[0] if isinstance(item, tuple) else item
                if not isinstance(doc, Document):
                    raise ValueError(f"Expected Document, got {type(doc)}")

                # Generate a unique identifier for the document
                doc_id = doc.metadata.get("id", hash(doc.page_content))
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0
                # Update score using the RRF formula
                fused_scores[doc_id] += 1 / (rank + self.k)

        # Sort documents by their RRF scores in descending order
        reranked_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        # Retrieve the original document objects for the top_n results
        doc_map = {doc.metadata.get("id", hash(doc.page_content)): doc for rl in ranked_lists for doc in rl if
                   isinstance(doc, Document)}
        top_documents = [
            doc_map[doc_id].copy(update={"metadata": {**doc_map[doc_id].metadata, "relevance_score": score}})
            for doc_id, score in reranked_docs[:self.top_n]
        ]

        return top_documents
