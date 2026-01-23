"""Async embedding vectorizer adapter for Redis middleware.

This module provides an async wrapper for synchronous embedding functions,
following the pattern used in production deployments.
"""

import asyncio
from typing import Any, Callable, List, Optional, Union


class AsyncEmbeddingVectorizer:
    """Async wrapper for synchronous embedding functions.

    This class wraps synchronous embedding functions (like those from
    langchain-openai) for use in async contexts using asyncio.to_thread().

    Example:
        ```python
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings()
        vectorizer = AsyncEmbeddingVectorizer(
            embed_fn=embeddings.embed_documents,
            dims=1536  # OpenAI ada-002 dimensions
        )

        # Use in async context
        vectors = await vectorizer.aembed(["hello world"])
        ```
    """

    def __init__(
        self,
        embed_fn: Callable[[List[str]], List[List[float]]],
        dims: int,
        *,
        embed_query_fn: Optional[Callable[[str], List[float]]] = None,
    ) -> None:
        """Initialize the async vectorizer.

        Args:
            embed_fn: Synchronous function that embeds a list of texts.
            dims: Dimensionality of the embedding vectors.
            embed_query_fn: Optional separate function for single query embedding.
                If not provided, embed_fn is used with a single-item list.
        """
        self._embed_fn = embed_fn
        self._embed_query_fn = embed_query_fn
        self._dims = dims

    @property
    def dims(self) -> int:
        """Return the embedding dimensions."""
        return self._dims

    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Synchronously embed texts.

        Args:
            texts: A single text or list of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if isinstance(texts, str):
            texts = [texts]
        return self._embed_fn(texts)

    async def aembed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Asynchronously embed texts using asyncio.to_thread().

        Args:
            texts: A single text or list of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if isinstance(texts, str):
            texts = [texts]
        return await asyncio.to_thread(self._embed_fn, texts)

    def embed_query(self, text: str) -> List[float]:
        """Synchronously embed a single query text.

        Args:
            text: The query text to embed.

        Returns:
            The embedding vector.
        """
        if self._embed_query_fn is not None:
            return self._embed_query_fn(text)
        return self._embed_fn([text])[0]

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronously embed a single query text.

        Args:
            text: The query text to embed.

        Returns:
            The embedding vector.
        """
        if self._embed_query_fn is not None:
            return await asyncio.to_thread(self._embed_query_fn, text)
        result = await asyncio.to_thread(self._embed_fn, [text])
        return result[0]


def create_vectorizer_from_langchain(
    embeddings: Any, *, dims: Optional[int] = None
) -> AsyncEmbeddingVectorizer:
    """Create an AsyncEmbeddingVectorizer from a LangChain Embeddings object.

    Args:
        embeddings: A LangChain Embeddings object (e.g., OpenAIEmbeddings).
        dims: Optional dimensions override. If not provided, will attempt
            to determine from the embeddings object or use a default.

    Returns:
        An AsyncEmbeddingVectorizer wrapping the LangChain embeddings.

    Example:
        ```python
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings()
        vectorizer = create_vectorizer_from_langchain(embeddings, dims=1536)
        ```
    """
    # Try to get dimensions from the embeddings object
    if dims is None:
        # Common attribute names for dimensions
        for attr in ["dimensions", "dims", "embedding_dim", "dimension"]:
            if hasattr(embeddings, attr):
                dims = getattr(embeddings, attr)
                break

    # If still None, try to infer from model name
    if dims is None:
        model = getattr(embeddings, "model", None) or getattr(
            embeddings, "model_name", None
        )
        if model:
            # Common OpenAI model dimensions
            model_dims = {
                "text-embedding-ada-002": 1536,
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
            }
            dims = model_dims.get(model)

    if dims is None:
        raise ValueError(
            "Could not determine embedding dimensions. "
            "Please provide dims parameter explicitly."
        )

    # Get the embedding functions
    embed_fn = embeddings.embed_documents
    embed_query_fn = getattr(embeddings, "embed_query", None)

    return AsyncEmbeddingVectorizer(
        embed_fn=embed_fn,
        dims=dims,
        embed_query_fn=embed_query_fn,
    )
