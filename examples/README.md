# Redis Notebooks for LangGraph

This directory contains Jupyter notebooks demonstrating the usage of Redis with LangGraph.

## Running Notebooks with Docker

To run these notebooks using Docker (recommended for consistent environment):

1. Ensure you have Docker and Docker Compose installed on your system.
2. Navigate to this directory (`examples`) in your terminal.
3. Run the following command:

   ```bash
   docker compose up
   ```

4. Look for a URL in the console output that starts with `http://127.0.0.1:8888/tree`. Open this URL in your web browser
   to access Jupyter Notebook.
5. You can now run the notebooks with all dependencies pre-installed.

Note:

- The first time you run this, it may take a few minutes to build the Docker image.
- When running with Docker Compose, the local library code from `../` (parent directory) is automatically mounted and installed,
  allowing you to test changes to the library immediately without rebuilding.
- If running the Docker image standalone (without docker-compose), it will install the library from PyPI instead.

To stop the Docker containers, use Ctrl+C in the terminal where you ran `docker compose up`, then run:

```bash
docker compose down
```

## Running Notebooks Locally

If you prefer to run these notebooks locally without Docker:

1. Make sure you have Redis running locally or accessible from your machine.
2. Install the required dependencies:

   ```bash
   pip install langgraph-checkpoint-redis
   pip install langgraph>=0.3.0
   pip install jupyter redis>=5.2.1 redisvl>=0.11.0
   pip install langchain-openai langchain-anthropic
   pip install python-ulid "httpx>=0.24.0,<1.0.0"

   # Some notebooks may require additional packages, which will be installed
   # within the notebooks themselves when needed
   ```

3. Set the appropriate Redis connection string in the notebooks.
4. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Navigate to the notebook you want to run and open it.

## Notebook Contents

- `persistence-functional.ipynb`: Demonstrates the usage of `RedisSaver` and functional persistence patterns with
  LangGraph.
- `create-react-agent-memory.ipynb`: Shows how to create an agent with persistent memory using Redis.
- `cross-thread-persistence.ipynb`: Demonstrates cross-thread persistence capabilities with Redis.
- `cross-thread-persistence-functional.ipynb`: Shows functional cross-thread persistence patterns with Redis.
- `create-react-agent-manage-message-history.ipynb`: Shows how to manage conversation history in a ReAct agent with
  Redis.
- `subgraph-persistence.ipynb`: Demonstrates persistence with subgraphs using Redis.
- `subgraphs-manage-state.ipynb`: Shows how to manage state in subgraphs with Redis.
- `create-react-agent-hitl.ipynb`: Demonstrates human-in-the-loop (HITL) capabilities with Redis.
- `human_in_the_loop/*.ipynb`: Demonstrates various human-in-the-loop interaction patterns with LangGraph and Redis.

All notebooks have been updated to use the Redis implementation instead of memory implementation, showcasing the proper
usage of Redis integration with LangGraph.

These notebooks are designed to work both within this Docker environment (using local package builds) and standalone (
using installed packages via pip).
