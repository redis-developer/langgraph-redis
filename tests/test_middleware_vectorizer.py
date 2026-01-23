"""Unit tests for the async embedding vectorizer."""

import pytest

from langgraph.middleware.redis.vectorizer import (
    AsyncEmbeddingVectorizer,
    create_vectorizer_from_langchain,
)


class TestAsyncEmbeddingVectorizer:
    """Tests for AsyncEmbeddingVectorizer class."""

    def test_init_with_embed_fn(self) -> None:
        """Test initialization with embed function."""

        def mock_embed(texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2, 0.3] for _ in texts]

        vectorizer = AsyncEmbeddingVectorizer(embed_fn=mock_embed, dims=3)
        assert vectorizer.dims == 3

    def test_dims_property(self) -> None:
        """Test that dims property returns correct value."""

        def mock_embed(texts: list[str]) -> list[list[float]]:
            return [[0.1] * 1536 for _ in texts]

        vectorizer = AsyncEmbeddingVectorizer(embed_fn=mock_embed, dims=1536)
        assert vectorizer.dims == 1536

    def test_embed_single_text(self) -> None:
        """Test synchronous embedding of a single text."""

        def mock_embed(texts: list[str]) -> list[list[float]]:
            return [[float(len(t))] for t in texts]

        vectorizer = AsyncEmbeddingVectorizer(embed_fn=mock_embed, dims=1)
        result = vectorizer.embed("hello")
        assert result == [[5.0]]

    def test_embed_multiple_texts(self) -> None:
        """Test synchronous embedding of multiple texts."""

        def mock_embed(texts: list[str]) -> list[list[float]]:
            return [[float(len(t)), float(len(t) * 2)] for t in texts]

        vectorizer = AsyncEmbeddingVectorizer(embed_fn=mock_embed, dims=2)
        result = vectorizer.embed(["hi", "hello"])
        assert result == [[2.0, 4.0], [5.0, 10.0]]

    def test_embed_query_without_custom_fn(self) -> None:
        """Test embed_query uses embed_fn when no custom query fn provided."""

        def mock_embed(texts: list[str]) -> list[list[float]]:
            return [[1.0, 2.0, 3.0] for _ in texts]

        vectorizer = AsyncEmbeddingVectorizer(embed_fn=mock_embed, dims=3)
        result = vectorizer.embed_query("test query")
        assert result == [1.0, 2.0, 3.0]

    def test_embed_query_with_custom_fn(self) -> None:
        """Test embed_query uses custom function when provided."""

        def mock_embed(texts: list[str]) -> list[list[float]]:
            return [[1.0, 2.0] for _ in texts]

        def mock_embed_query(text: str) -> list[float]:
            return [9.0, 9.0]  # Different result

        vectorizer = AsyncEmbeddingVectorizer(
            embed_fn=mock_embed, dims=2, embed_query_fn=mock_embed_query
        )
        result = vectorizer.embed_query("test")
        assert result == [9.0, 9.0]

    @pytest.mark.asyncio
    async def test_aembed_single_text(self) -> None:
        """Test async embedding of a single text."""

        def mock_embed(texts: list[str]) -> list[list[float]]:
            return [[float(len(t))] for t in texts]

        vectorizer = AsyncEmbeddingVectorizer(embed_fn=mock_embed, dims=1)
        result = await vectorizer.aembed("world")
        assert result == [[5.0]]

    @pytest.mark.asyncio
    async def test_aembed_multiple_texts(self) -> None:
        """Test async embedding of multiple texts."""

        def mock_embed(texts: list[str]) -> list[list[float]]:
            return [[1.0] * len(texts)]

        vectorizer = AsyncEmbeddingVectorizer(embed_fn=mock_embed, dims=1)
        result = await vectorizer.aembed(["a", "b", "c"])
        # The mock returns a list with one vector containing 3 elements
        assert result == [[1.0, 1.0, 1.0]]

    @pytest.mark.asyncio
    async def test_aembed_query_without_custom_fn(self) -> None:
        """Test async embed_query uses embed_fn when no custom fn."""

        def mock_embed(texts: list[str]) -> list[list[float]]:
            return [[3.14, 2.71] for _ in texts]

        vectorizer = AsyncEmbeddingVectorizer(embed_fn=mock_embed, dims=2)
        result = await vectorizer.aembed_query("test")
        assert result == [3.14, 2.71]

    @pytest.mark.asyncio
    async def test_aembed_query_with_custom_fn(self) -> None:
        """Test async embed_query uses custom function when provided."""

        def mock_embed(texts: list[str]) -> list[list[float]]:
            return [[1.0, 2.0] for _ in texts]

        def mock_embed_query(text: str) -> list[float]:
            return [7.0, 8.0]

        vectorizer = AsyncEmbeddingVectorizer(
            embed_fn=mock_embed, dims=2, embed_query_fn=mock_embed_query
        )
        result = await vectorizer.aembed_query("test")
        assert result == [7.0, 8.0]


class TestCreateVectorizerFromLangchain:
    """Tests for create_vectorizer_from_langchain function."""

    def test_with_explicit_dims(self) -> None:
        """Test creating vectorizer with explicit dimensions."""

        class MockEmbeddings:
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [[0.1] * 768 for _ in texts]

        embeddings = MockEmbeddings()
        vectorizer = create_vectorizer_from_langchain(embeddings, dims=768)
        assert vectorizer.dims == 768

    def test_with_dimensions_attribute(self) -> None:
        """Test creating vectorizer when embeddings has dimensions attr."""

        class MockEmbeddings:
            dimensions = 1024

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [[0.1] * 1024 for _ in texts]

        embeddings = MockEmbeddings()
        vectorizer = create_vectorizer_from_langchain(embeddings)
        assert vectorizer.dims == 1024

    def test_with_dims_attribute(self) -> None:
        """Test creating vectorizer when embeddings has dims attr."""

        class MockEmbeddings:
            dims = 512

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [[0.1] * 512 for _ in texts]

        embeddings = MockEmbeddings()
        vectorizer = create_vectorizer_from_langchain(embeddings)
        assert vectorizer.dims == 512

    def test_with_model_name_openai_ada(self) -> None:
        """Test dimension inference from OpenAI ada-002 model name."""

        class MockEmbeddings:
            model = "text-embedding-ada-002"

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [[0.1] * 1536 for _ in texts]

        embeddings = MockEmbeddings()
        vectorizer = create_vectorizer_from_langchain(embeddings)
        assert vectorizer.dims == 1536

    def test_with_model_name_3_large(self) -> None:
        """Test dimension inference from OpenAI embedding-3-large model."""

        class MockEmbeddings:
            model = "text-embedding-3-large"

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [[0.1] * 3072 for _ in texts]

        embeddings = MockEmbeddings()
        vectorizer = create_vectorizer_from_langchain(embeddings)
        assert vectorizer.dims == 3072

    def test_with_model_name_attribute(self) -> None:
        """Test dimension inference using model_name attribute."""

        class MockEmbeddings:
            model_name = "text-embedding-3-small"

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [[0.1] * 1536 for _ in texts]

        embeddings = MockEmbeddings()
        vectorizer = create_vectorizer_from_langchain(embeddings)
        assert vectorizer.dims == 1536

    def test_raises_without_dims(self) -> None:
        """Test that ValueError is raised when dims cannot be determined."""

        class MockEmbeddings:
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [[0.1] for _ in texts]

        embeddings = MockEmbeddings()
        with pytest.raises(
            ValueError, match="Could not determine embedding dimensions"
        ):
            create_vectorizer_from_langchain(embeddings)

    def test_with_embed_query_method(self) -> None:
        """Test that embed_query method is properly wrapped."""

        class MockEmbeddings:
            dimensions = 3

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [[1.0, 2.0, 3.0] for _ in texts]

            def embed_query(self, text: str) -> list[float]:
                return [4.0, 5.0, 6.0]

        embeddings = MockEmbeddings()
        vectorizer = create_vectorizer_from_langchain(embeddings)

        # Check that embed_query is used
        result = vectorizer.embed_query("test")
        assert result == [4.0, 5.0, 6.0]

    def test_functional_embedding(self) -> None:
        """Test that the created vectorizer actually embeds correctly."""

        class MockEmbeddings:
            dimensions = 2

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [[float(len(t)), float(len(t) * 2)] for t in texts]

        embeddings = MockEmbeddings()
        vectorizer = create_vectorizer_from_langchain(embeddings)

        result = vectorizer.embed(["hi", "hello"])
        assert result == [[2.0, 4.0], [5.0, 10.0]]
