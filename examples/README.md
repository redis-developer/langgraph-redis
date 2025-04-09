# Redis Notebooks for LangGraph

This directory contains Jupyter notebooks demonstrating the usage of the Redis with LangGraph.

## Running Notebooks with Docker

To run these notebooks using the local development versions of LangChain and the Redis partner package:

1. Ensure you have Docker and Docker Compose installed on your system.
2. Navigate to this directory (`examples`) in your terminal.
3. Run the following command:
   ```bash
   docker compose up
   ```
4. Look for a URL in the console output that starts with `http://127.0.0.1:8888/tree`. Open this URL in your web browser to access Jupyter Notebook.
5. You can now run the notebooks, which will use the local development versions of LangChain and the Redis partner package.

Note: The first time you run this, it may take a few minutes to build the Docker image.

To stop the Docker containers, use Ctrl+C in the terminal where you ran `docker compose up`, then run:
```bash
docker compose down
```

## Notebook Contents

- `persistence_redis.ipynb`: Demonstrates the usage of `RedisSaver` and `AsyncRedisSaver` checkpoint savers with LangGraph.

These notebooks are designed to work both within this Docker environment (using local package builds) and standalone (using installed packages via pip).
