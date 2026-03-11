Middleware
==========

Redis-backed middleware for LangGraph agent workflows, including semantic caching, tool result caching, conversation memory, and semantic routing.

Middleware Implementations
--------------------------

.. autoclass:: langgraph.middleware.redis.SemanticCacheMiddleware
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: langgraph.middleware.redis.ToolResultCacheMiddleware
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: langgraph.middleware.redis.ConversationMemoryMiddleware
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: langgraph.middleware.redis.SemanticRouterMiddleware
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: langgraph.middleware.redis.MiddlewareStack
   :members:
   :undoc-members:
   :show-inheritance:

Base Classes
------------

.. autoclass:: langgraph.middleware.redis.aio.AsyncRedisMiddleware
   :members:
   :undoc-members:
   :show-inheritance:

Configuration Types
-------------------

.. autoclass:: langgraph.middleware.redis.types.SemanticCacheConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: langgraph.middleware.redis.types.ToolCacheConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: langgraph.middleware.redis.types.ConversationMemoryConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: langgraph.middleware.redis.types.SemanticRouterConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: langgraph.middleware.redis.types.MiddlewareConfig
   :members:
   :undoc-members:
   :show-inheritance:
