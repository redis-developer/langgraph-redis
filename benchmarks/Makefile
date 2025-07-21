.PHONY: help install benchmark plots clean all

help:
	@echo "LangGraph Redis Checkpointer Benchmarks"
	@echo ""
	@echo "Available targets:"
	@echo "  install    - Install benchmark dependencies"
	@echo "  benchmark  - Run comprehensive checkpointer benchmark"
	@echo "  plots      - Generate performance plots from results"
	@echo "  all        - Run benchmark and generate plots"
	@echo "  clean      - Remove generated reports and plots"
	@echo ""
	@echo "Prerequisites:"
	@echo "  - Docker (for TestContainers)"
	@echo "  - Optional: Redis Stack for local Redis testing"

install:
	poetry install

benchmark:
	@echo "🚀 Running comprehensive checkpointer benchmark..."
	poetry run python checkpointer_benchmark.py

plots:
	@echo "📊 Generating performance plots..."
	poetry run python create_plots.py

all: benchmark plots
	@echo "✅ Benchmark and plots completed"

clean:
	@echo "🧹 Cleaning up reports and plots..."
	rm -rf reports/
	rm -f *.png *.jpg *.svg