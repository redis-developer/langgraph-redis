"""Async embedding vectorizer adapter for Redis middleware.

This module provides vectorizer adapters for Redis middleware:

1. AsyncEmbeddingVectorizer: A simple async wrapper for embedding functions
2. LangChainVectorizer: A redisvl-compatible BaseVectorizer that wraps LangChain embeddings

The LangChainVectorizer is the recommended approach for using LangChain embeddings
(OpenAI, Azure OpenAI, etc.) with SemanticCache and SemanticCacheMiddleware.
"""

import asyncio
from typing import Any, Callable, List, Optional, Union

from redisvl.utils.vectorize.base import BaseVectorizer


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


class LangChainVectorizer(BaseVectorizer):
    """A redisvl-compatible vectorizer that wraps LangChain Embeddings.

    This class bridges LangChain embeddings (OpenAI, Azure OpenAI, HuggingFace, etc.)
    with redisvl's SemanticCache by implementing the BaseVectorizer interface.

    Unlike AsyncEmbeddingVectorizer, this class properly extends redisvl's
    BaseVectorizer (a Pydantic model), making it compatible with SemanticCache's
    type validation.

    Example:
        ```python
        from langchain_openai import OpenAIEmbeddings
        from langgraph.middleware.redis import LangChainVectorizer, SemanticCacheConfig

        # Create LangChain embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # Wrap in LangChainVectorizer for redisvl compatibility
        vectorizer = LangChainVectorizer(embeddings=embeddings, dims=1536)

        # Use with SemanticCacheMiddleware
        config = SemanticCacheConfig(
            redis_url="redis://localhost:6379",
            vectorizer=vectorizer,
        )
        ```

    For Azure OpenAI:
        ```python
        from langchain_openai import AzureOpenAIEmbeddings

        embeddings = AzureOpenAIEmbeddings(
            azure_deployment="your-embedding-deployment",
            openai_api_version="2024-02-15-preview",
        )
        vectorizer = LangChainVectorizer(embeddings=embeddings, dims=1536)
        ```
    """

    # Store the LangChain embeddings object (excluded from Pydantic serialization)
    _embeddings: Any = None

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, embeddings: Any, dims: int, **kwargs: Any) -> None:
        """Initialize the LangChain vectorizer.

        Args:
            embeddings: A LangChain Embeddings object (e.g., OpenAIEmbeddings,
                AzureOpenAIEmbeddings, HuggingFaceEmbeddings).
            dims: The dimensionality of the embedding vectors.
            **kwargs: Additional arguments passed to BaseVectorizer.
        """
        # Initialize BaseVectorizer with required fields
        super().__init__(model="langchain", dims=dims, **kwargs)
        # Store embeddings as private attribute (not a Pydantic field)
        object.__setattr__(self, "_embeddings", embeddings)

    def embed(
        self,
        text: str,
        preprocess: Optional[Callable] = None,
        as_buffer: bool = False,
        **kwargs: Any,
    ) -> Union[List[float], bytes]:
        """Embed a single text using LangChain embeddings.

        Args:
            text: The text to embed.
            preprocess: Optional preprocessing function (applied if provided).
            as_buffer: If True, return as bytes buffer.
            **kwargs: Additional arguments (ignored).

        Returns:
            The embedding vector as a list of floats, or bytes if as_buffer=True.
        """
        if preprocess is not None:
            text = preprocess(text)

        # Use embed_query for single text (more efficient for some providers)
        if hasattr(self._embeddings, "embed_query"):
            embedding = self._embeddings.embed_query(text)
        else:
            embedding = self._embeddings.embed_documents([text])[0]

        if as_buffer:
            return self._process_embedding(embedding, as_buffer=True, dtype=self.dtype)
        return embedding

    def embed_many(
        self,
        texts: List[str],
        preprocess: Optional[Callable] = None,
        batch_size: int = 1000,
        as_buffer: bool = False,
        **kwargs: Any,
    ) -> Union[List[List[float]], List[bytes]]:
        """Embed multiple texts using LangChain embeddings.

        Args:
            texts: List of texts to embed.
            preprocess: Optional preprocessing function (applied to each text).
            batch_size: Batch size for processing (used for batching).
            as_buffer: If True, return as bytes buffers.
            **kwargs: Additional arguments (ignored).

        Returns:
            List of embedding vectors, or list of bytes if as_buffer=True.
        """
        all_embeddings: List[List[float]] = []

        for batch in self.batchify(texts, batch_size, preprocess):
            embeddings = self._embeddings.embed_documents(batch)
            if as_buffer:
                embeddings = [
                    self._process_embedding(e, as_buffer=True, dtype=self.dtype)
                    for e in embeddings
                ]
            all_embeddings.extend(embeddings)

        return all_embeddings

    async def aembed(
        self,
        text: str,
        preprocess: Optional[Callable] = None,
        as_buffer: bool = False,
        **kwargs: Any,
    ) -> Union[List[float], bytes]:
        """Asynchronously embed a single text.

        Args:
            text: The text to embed.
            preprocess: Optional preprocessing function.
            as_buffer: If True, return as bytes buffer.
            **kwargs: Additional arguments (ignored).

        Returns:
            The embedding vector as a list of floats, or bytes if as_buffer=True.
        """
        if preprocess is not None:
            text = preprocess(text)

        # Try async method first, fall back to sync with to_thread
        if hasattr(self._embeddings, "aembed_query"):
            embedding = await self._embeddings.aembed_query(text)
        elif hasattr(self._embeddings, "aembed_documents"):
            embeddings = await self._embeddings.aembed_documents([text])
            embedding = embeddings[0]
        else:
            # Fall back to sync method in thread pool
            embedding = await asyncio.to_thread(self.embed, text, None, False)

        if as_buffer:
            return self._process_embedding(embedding, as_buffer=True, dtype=self.dtype)
        return embedding

    async def aembed_many(
        self,
        texts: List[str],
        preprocess: Optional[Callable] = None,
        batch_size: int = 1000,
        as_buffer: bool = False,
        **kwargs: Any,
    ) -> Union[List[List[float]], List[bytes]]:
        """Asynchronously embed multiple texts.

        Args:
            texts: List of texts to embed.
            preprocess: Optional preprocessing function.
            batch_size: Batch size for processing.
            as_buffer: If True, return as bytes buffers.
            **kwargs: Additional arguments (ignored).

        Returns:
            List of embedding vectors, or list of bytes if as_buffer=True.
        """
        all_embeddings: List[List[float]] = []

        for batch in self.batchify(texts, batch_size, preprocess):
            # Try async method first
            if hasattr(self._embeddings, "aembed_documents"):
                embeddings = await self._embeddings.aembed_documents(batch)
            else:
                # Fall back to sync method in thread pool
                embeddings = await asyncio.to_thread(
                    self._embeddings.embed_documents, batch
                )

            if as_buffer:
                embeddings = [
                    self._process_embedding(e, as_buffer=True, dtype=self.dtype)
                    for e in embeddings
                ]
            all_embeddings.extend(embeddings)

        return all_embeddings


def create_langchain_vectorizer(
    embeddings: Any, dims: Optional[int] = None
) -> LangChainVectorizer:
    """Create a LangChainVectorizer from a LangChain Embeddings object.

    This is the recommended way to use LangChain embeddings with SemanticCache
    and SemanticCacheMiddleware. The returned vectorizer is fully compatible
    with redisvl's BaseVectorizer interface.

    Args:
        embeddings: A LangChain Embeddings object (e.g., OpenAIEmbeddings,
            AzureOpenAIEmbeddings, HuggingFaceEmbeddings).
        dims: The embedding dimensions. If not provided, will attempt to
            infer from the embeddings object or known model dimensions.

    Returns:
        A LangChainVectorizer wrapping the embeddings.

    Raises:
        ValueError: If dimensions cannot be determined.

    Example:
        ```python
        from langchain_openai import OpenAIEmbeddings
        from langgraph.middleware.redis import create_langchain_vectorizer

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorizer = create_langchain_vectorizer(embeddings, dims=1536)
        ```
    """
    # Try to get dimensions from the embeddings object
    if dims is None:
        # Common attribute names for dimensions
        for attr in ["dimensions", "dims", "embedding_dim", "dimension"]:
            if hasattr(embeddings, attr):
                val = getattr(embeddings, attr)
                if val is not None:
                    dims = val
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

    return LangChainVectorizer(embeddings=embeddings, dims=dims)
