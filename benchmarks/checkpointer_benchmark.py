#!/usr/bin/env python3
"""
Comprehensive Checkpointer Performance Benchmark

This benchmark:
1. Tests all available checkpointer implementations (regular and shallow variants)
2. Uses fanout-to-subgraph pattern with realistic workload
3. Scales from 100 to 900 parallel executions
4. Generates individual JSON reports for each implementation
5. Creates performance plots for visual comparison
6. Includes both fanout scaling and individual operation benchmarks
"""

import asyncio
import time
import datetime
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Annotated
from dataclasses import dataclass, asdict
import operator
import importlib.metadata
import statistics

# Plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
    print("✅ Plotting libraries available")
except ImportError:
    PLOTTING_AVAILABLE = False
    print("❌ No plotting - install matplotlib and seaborn")

# LangGraph imports
from langgraph.constants import START, Send
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# Try to import checkpointers - track what actually works
available_checkpointers = {}
failed_imports = {}

# SQLite - regular and shallow
try:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    # Check if shallow version exists
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncShallowSqliteSaver
        available_checkpointers['sqlite'] = AsyncSqliteSaver
        available_checkpointers['sqlite_shallow'] = AsyncShallowSqliteSaver
        print("✅ AsyncSqliteSaver imported")
        print("✅ AsyncShallowSqliteSaver imported")
    except ImportError:
        # Only regular version available
        available_checkpointers['sqlite'] = AsyncSqliteSaver
        print("✅ AsyncSqliteSaver imported")
        print("⚠️  AsyncShallowSqliteSaver not available")
except ImportError as e:
    failed_imports['sqlite'] = str(e)
    print(f"❌ AsyncSqliteSaver failed: {e}")


# MongoDB - regular and shallow
try:
    from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
    # Check if shallow version exists
    try:
        from langgraph.checkpoint.mongodb.aio import AsyncShallowMongoDBSaver
        available_checkpointers['mongodb'] = AsyncMongoDBSaver
        available_checkpointers['mongodb_shallow'] = AsyncShallowMongoDBSaver
        print("✅ AsyncMongoDBSaver imported")
        print("✅ AsyncShallowMongoDBSaver imported")
    except ImportError:
        # Only regular version available
        available_checkpointers['mongodb'] = AsyncMongoDBSaver
        print("✅ AsyncMongoDBSaver imported")
        print("⚠️  AsyncShallowMongoDBSaver not available")
except ImportError as e:
    failed_imports['mongodb'] = str(e)
    print(f"❌ AsyncMongoDBSaver failed: {e}")

# Redis - Both regular and shallow implementations
try:
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver
    from langgraph.checkpoint.redis.ashallow import AsyncShallowRedisSaver
    available_checkpointers['redis'] = AsyncRedisSaver
    available_checkpointers['redis_shallow'] = AsyncShallowRedisSaver
    print("✅ AsyncRedisSaver imported")
    print("✅ AsyncShallowRedisSaver imported")
except ImportError as e:
    failed_imports['redis'] = str(e)
    print(f"❌ Redis checkpointers failed: {e}")

# MySQL - regular and shallow
try:
    from langgraph.checkpoint.mysql.aio import AIOMySQLSaver
    # Check if shallow version exists
    try:
        from langgraph.checkpoint.mysql.aio import ShallowAIOMySQLSaver
        available_checkpointers['mysql'] = AIOMySQLSaver
        available_checkpointers['mysql_shallow'] = ShallowAIOMySQLSaver
        print("✅ AIOMySQLSaver imported")
        print("✅ ShallowAIOMySQLSaver imported")
    except ImportError:
        # Only regular version available
        available_checkpointers['mysql'] = AIOMySQLSaver
        print("✅ AIOMySQLSaver imported")
        print("⚠️  ShallowAIOMySQLSaver not available")
except ImportError as e:
    failed_imports['mysql'] = str(e)
    print(f"❌ MySQL checkpointers failed: {e}")

# TestContainers for isolated testing
try:
    from testcontainers.redis import RedisContainer
    from testcontainers.mongodb import MongoDbContainer
    from testcontainers.mysql import MySqlContainer
    CONTAINERS_AVAILABLE = True
    print("✅ TestContainers available")
except ImportError:
    CONTAINERS_AVAILABLE = False
    print("❌ TestContainers not available")


# fanout-to-subgraph pattern
class OverallState(dict):
    subjects: List[str]
    jokes: Annotated[List[str], operator.add]

class JokeInput(dict):
    subject: str

class JokeOutput(dict):
    jokes: List[str]

class JokeState(JokeInput, JokeOutput):
    pass


@dataclass
class BenchmarkResult:
    """Raw benchmark result."""
    implementation: str
    version: str
    scale: int
    execution_time_sec: float
    operations_per_sec: float
    memory_mb: float
    timestamp: str
    error_message: str = None
    success: bool = True


class CheckpointerBenchmark:
    """Clean benchmark with no prior contamination."""
    
    def __init__(self):
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results: List[BenchmarkResult] = []
        self.containers = {}
        self.checkpointers = {}
        
        # Create reports directory
        self.reports_dir = Path(f"reports/run-{self.timestamp}")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Reports will be saved to: {self.reports_dir}")
        
        # scaling scenario
        self.scales_to_test = [100, 200, 400, 600, 900]
    
    def get_version(self, implementation: str) -> str:
        """Get actual package version."""
        version_map = {
            'memory': 'langgraph-core',
            'sqlite': 'langgraph-checkpoint-sqlite',
            'sqlite_shallow': 'langgraph-checkpoint-sqlite',
            'mongodb': 'langgraph-checkpoint-mongodb',
            'mongodb_shallow': 'langgraph-checkpoint-mongodb',
            'redis': 'langgraph-checkpoint-redis',
            'redis_shallow': 'langgraph-checkpoint-redis',
            'mysql': 'langgraph-checkpoint-mysql',
            'mysql_shallow': 'langgraph-checkpoint-mysql'
        }
        
        try:
            package = version_map.get(implementation, 'unknown')
            return importlib.metadata.version(package)
        except:
            try:
                return importlib.metadata.version('langgraph')
            except:
                return "unknown"
    
    async def setup_checkpointers(self):
        """Setup only the checkpointers that actually work."""
        print("\n🔧 Setting up checkpointers...")
        
        # 1. Memory - always available
        self.checkpointers['memory'] = MemorySaver()
        print("✅ Memory checkpointer ready")
        
        # 2. Redis (regular) - if available
        if 'redis' in available_checkpointers and CONTAINERS_AVAILABLE:
            try:
                # Start Redis Stack container
                redis_container = RedisContainer("redis/redis-stack-server:latest")
                redis_container.start()
                self.containers['redis'] = redis_container
                
                host = redis_container.get_container_host_ip()
                port = redis_container.get_exposed_port(6379)
                redis_url = f"redis://{host}:{port}"
                
                # Setup async Redis checkpointer (regular)
                redis_context = AsyncRedisSaver.from_conn_string(redis_url)
                redis_checkpointer = await redis_context.__aenter__()
                
                self.checkpointers['redis'] = redis_checkpointer
                self.redis_context = redis_context  # Store for cleanup
                print("✅ Redis (regular) checkpointer ready")
            except Exception as e:
                print(f"❌ Redis setup failed: {e}")
        
        # 2b. Redis Shallow - if available (reuse same container)
        if 'redis_shallow' in available_checkpointers and CONTAINERS_AVAILABLE and 'redis' in self.containers:
            try:
                # Reuse the same Redis container
                redis_container = self.containers['redis']
                host = redis_container.get_container_host_ip()
                port = redis_container.get_exposed_port(6379)
                redis_url = f"redis://{host}:{port}"
                
                # Setup async Redis shallow checkpointer
                redis_shallow_context = AsyncShallowRedisSaver.from_conn_string(redis_url)
                redis_shallow_checkpointer = await redis_shallow_context.__aenter__()
                
                self.checkpointers['redis_shallow'] = redis_shallow_checkpointer
                self.redis_shallow_context = redis_shallow_context  # Store for cleanup
                print("✅ Redis (shallow) checkpointer ready")
            except Exception as e:
                print(f"❌ Redis shallow setup failed: {e}")
        
        # 3. SQLite (regular) - if available  
        if 'sqlite' in available_checkpointers:
            try:
                sqlite_context = AsyncSqliteSaver.from_conn_string(":memory:")
                sqlite_checkpointer = await sqlite_context.__aenter__()
                await sqlite_checkpointer.setup()
                
                self.checkpointers['sqlite'] = sqlite_checkpointer  
                self.sqlite_context = sqlite_context
                print("✅ SQLite (regular) checkpointer ready")
            except Exception as e:
                print(f"❌ SQLite setup failed: {e}")
        
        # 3b. SQLite Shallow - if available
        if 'sqlite_shallow' in available_checkpointers:
            try:
                sqlite_shallow_context = AsyncShallowSqliteSaver.from_conn_string(":memory:")
                sqlite_shallow_checkpointer = await sqlite_shallow_context.__aenter__()
                await sqlite_shallow_checkpointer.setup()
                
                self.checkpointers['sqlite_shallow'] = sqlite_shallow_checkpointer  
                self.sqlite_shallow_context = sqlite_shallow_context
                print("✅ SQLite (shallow) checkpointer ready")
            except Exception as e:
                print(f"❌ SQLite shallow setup failed: {e}")
        
        
        # 5. MongoDB (regular) - if available
        if 'mongodb' in available_checkpointers and CONTAINERS_AVAILABLE:
            try:
                mongo_container = MongoDbContainer("mongo:7")
                mongo_container.start()
                self.containers['mongodb'] = mongo_container
                
                mongo_url = mongo_container.get_connection_url()
                
                mongo_context = AsyncMongoDBSaver.from_conn_string(
                    mongo_url,
                    db_name=f"benchmark_{self.timestamp}"
                )
                mongo_checkpointer = await mongo_context.__aenter__()
                
                self.checkpointers['mongodb'] = mongo_checkpointer
                self.mongodb_context = mongo_context
                print("✅ MongoDB (regular) checkpointer ready")
            except Exception as e:
                print(f"❌ MongoDB setup failed: {e}")
        
        # 5b. MongoDB Shallow - if available (reuse same container)
        if 'mongodb_shallow' in available_checkpointers and CONTAINERS_AVAILABLE and 'mongodb' in self.containers:
            try:
                # Reuse the same MongoDB container
                mongo_container = self.containers['mongodb']
                mongo_url = mongo_container.get_connection_url()
                
                mongo_shallow_context = AsyncShallowMongoDBSaver.from_conn_string(
                    mongo_url,
                    db_name=f"benchmark_shallow_{self.timestamp}"
                )
                mongo_shallow_checkpointer = await mongo_shallow_context.__aenter__()
                
                self.checkpointers['mongodb_shallow'] = mongo_shallow_checkpointer
                self.mongodb_shallow_context = mongo_shallow_context
                print("✅ MongoDB (shallow) checkpointer ready")
            except Exception as e:
                print(f"❌ MongoDB shallow setup failed: {e}")
        
        # 6. MySQL (regular) - if available
        if 'mysql' in available_checkpointers and CONTAINERS_AVAILABLE:
            try:
                mysql_container = MySqlContainer("mysql:8")
                mysql_container.start()
                self.containers['mysql'] = mysql_container
                
                # Get connection URL and build MySQL connection string
                mysql_url = f"mysql://{mysql_container.username}:{mysql_container.password}@{mysql_container.get_container_host_ip()}:{mysql_container.get_exposed_port(3306)}/{mysql_container.dbname}"
                
                mysql_context = AIOMySQLSaver.from_conn_string(mysql_url)
                mysql_checkpointer = await mysql_context.__aenter__()
                await mysql_checkpointer.setup()
                
                self.checkpointers['mysql'] = mysql_checkpointer
                self.mysql_context = mysql_context
                print("✅ MySQL (regular) checkpointer ready")
            except Exception as e:
                print(f"❌ MySQL setup failed: {e}")
        
        # 6b. MySQL Shallow - if available (use separate container to avoid schema conflicts)
        if 'mysql_shallow' in available_checkpointers and CONTAINERS_AVAILABLE:
            try:
                # Use a separate MySQL container to avoid schema conflicts with regular MySQL
                mysql_shallow_container = MySqlContainer("mysql:8")
                mysql_shallow_container.start()
                self.containers['mysql_shallow'] = mysql_shallow_container
                
                mysql_shallow_url = f"mysql://{mysql_shallow_container.username}:{mysql_shallow_container.password}@{mysql_shallow_container.get_container_host_ip()}:{mysql_shallow_container.get_exposed_port(3306)}/{mysql_shallow_container.dbname}"
                
                mysql_shallow_context = ShallowAIOMySQLSaver.from_conn_string(mysql_shallow_url)
                mysql_shallow_checkpointer = await mysql_shallow_context.__aenter__()
                await mysql_shallow_checkpointer.setup()
                
                self.checkpointers['mysql_shallow'] = mysql_shallow_checkpointer
                self.mysql_shallow_context = mysql_shallow_context
                print("✅ MySQL (shallow) checkpointer ready")
            except Exception as e:
                print(f"❌ MySQL shallow setup failed: {e}")
        
        print(f"\n📊 Ready to test {len(self.checkpointers)} implementations:")
        for impl in self.checkpointers.keys():
            version = self.get_version(impl)
            print(f"   - {impl}: {version}")
    
    async def create_fanout_graph(self):
        """Create fanout-to-subgraph pattern."""
        
        # Subgraph nodes 
        async def edit(state: JokeInput):
            subject = state["subject"]
            return {"subject": f"{subject}, and cats"}

        async def generate(state: JokeInput):
            return {"jokes": [f"Joke about the year {state['subject']}"]}

        async def bump(state: JokeOutput):
            return {"jokes": [state["jokes"][0] + " and more"]}

        async def bump_loop(state: JokeOutput):
            return END if state["jokes"][0].endswith(" and more" * 10) else "bump"

        # Build subgraph
        subgraph = StateGraph(JokeState, input=JokeInput, output=JokeOutput)
        subgraph.add_node("edit", edit)
        subgraph.add_node("generate", generate) 
        subgraph.add_node("bump", bump)
        subgraph.set_entry_point("edit")
        subgraph.add_edge("edit", "generate")
        subgraph.add_edge("generate", "bump")
        subgraph.add_conditional_edges("bump", bump_loop)
        subgraph.set_finish_point("generate")
        compiled_subgraph = subgraph.compile()

        # Parent graph with fanout
        async def fanout(state: OverallState):
            return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

        parent_graph = StateGraph(OverallState)
        parent_graph.add_node("generate_joke", compiled_subgraph)
        parent_graph.add_conditional_edges(START, fanout)
        parent_graph.add_edge("generate_joke", END)
        
        return parent_graph
    
    async def run_scale_test(self, impl_name: str, checkpointer, scale: int) -> BenchmarkResult:
        """Run one scale test."""
        print(f"  🔄 Testing {impl_name} at scale {scale}...")
        
        try:
            # Generate test data - years
            years = [str(2025 - i) for i in range(scale)]
            input_data = {"subjects": years}
            
            # Create graph
            graph_builder = await self.create_fanout_graph()
            graph = graph_builder.compile(checkpointer=checkpointer)
            
            config = {"configurable": {"thread_id": f"{impl_name}_{scale}_{self.timestamp}"}}
            
            # All checkpointers should handle the fanout pattern correctly
            
            # Measure execution time
            import psutil
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024
            
            start_time = time.monotonic()
            
            # Execute fanout pattern  
            results = []
            async for chunk in graph.astream(input_data, config=config):
                results.append(chunk)
            
            end_time = time.monotonic()
            memory_after = process.memory_info().rss / 1024 / 1024
            
            # Validate we got expected results
            if len(results) != scale:
                raise ValueError(f"Expected {scale} results, got {len(results)}")
            
            execution_time = end_time - start_time
            ops_per_sec = scale / execution_time if execution_time > 0 else 0
            memory_used = memory_after - memory_before
            
            result = BenchmarkResult(
                implementation=impl_name,
                version=self.get_version(impl_name),
                scale=scale,
                execution_time_sec=execution_time,
                operations_per_sec=ops_per_sec,
                memory_mb=memory_used,
                timestamp=self.timestamp,
                success=True
            )
            
            print(f"    ✅ {execution_time:.2f}s ({ops_per_sec:.0f} ops/sec)")
            return result
            
        except Exception as e:
            error_msg = str(e)
            print(f"    ❌ Failed: {error_msg}")
            return BenchmarkResult(
                implementation=impl_name,
                version=self.get_version(impl_name),
                scale=scale,
                execution_time_sec=0,
                operations_per_sec=0,
                memory_mb=0,
                timestamp=self.timestamp,
                error_message=error_msg,
                success=False
            )
    
    async def run_benchmark(self):
        """Run the comprehensive benchmark."""
        print(f"\n🎯 Running Comprehensive Checkpointer Benchmark")
        
        # 1. Fanout-to-subgraph scaling test
        await self.run_fanout_benchmark()
        
        # 2. Individual operation benchmarks
        await self.run_individual_operation_benchmarks()
    
    async def run_fanout_benchmark(self):
        """Run fanout-to-subgraph scaling test."""
        print(f"\n📊 Running Fanout-to-Subgraph Benchmark")
        print(f"   Testing scales: {self.scales_to_test}")
        
        for scale in self.scales_to_test:
            print(f"\n📈 Scale: {scale} parallel executions")
            
            for impl_name, checkpointer in self.checkpointers.items():
                result = await self.run_scale_test(impl_name, checkpointer, scale)
                self.results.append(result)
    
    async def run_individual_operation_benchmarks(self):
        """Run individual checkpointer operation benchmarks."""
        print(f"\n📊 Running Individual Operation Benchmarks")
        
        scenarios = [
            {
                "name": "Single Checkpoint GET",
                "operation": "get_checkpoint", 
                "channels": 10,
                "checkpoints": 1,
                "method": self.benchmark_single_get
            },
            {
                "name": "Channel Value Loading",
                "operation": "get_channel_values",
                "channels": 20, 
                "checkpoints": 1,
                "method": self.benchmark_channel_loading
            },
            {
                "name": "List Operations",
                "operation": "list_checkpoints",
                "channels": 15,
                "checkpoints": 25,
                "method": self.benchmark_list_operations
            }
        ]
        
        for scenario in scenarios:
            print(f"\n🔍 Testing {scenario['name']}...")
            
            for impl_name, checkpointer in self.checkpointers.items():
                try:
                    print(f"  🔄 Testing {impl_name}...")
                    
                    # Run multiple iterations
                    runs = []
                    for _ in range(5):  # 5 runs for statistical significance
                        timing = await scenario["method"](checkpointer, scenario["channels"], scenario.get("checkpoints", 1))
                        runs.append(timing)
                    
                    mean_time = statistics.mean(runs) 
                    p95_time = max(runs)
                    p99_time = max(runs)
                    
                    # Memory usage
                    import psutil
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    
                    result = BenchmarkResult(
                        implementation=impl_name,
                        version=self.get_version(impl_name),
                        scale=scenario.get("checkpoints", 1),
                        execution_time_sec=mean_time / 1000,  # Convert ms to seconds
                        operations_per_sec=1000 / mean_time if mean_time > 0 else 0,
                        memory_mb=memory_mb,
                        timestamp=self.timestamp,
                        success=True,
                        # Add operation-specific metadata
                        error_message=f"operation={scenario['operation']};channels={scenario['channels']}"
                    )
                    
                    self.results.append(result)
                    print(f"    ✅ {mean_time:.2f}ms avg")
                    
                except Exception as e:
                    error_msg = str(e)
                    print(f"    ❌ Failed: {error_msg}")
                    result = BenchmarkResult(
                        implementation=impl_name,
                        version=self.get_version(impl_name),
                        scale=scenario.get("checkpoints", 1),
                        execution_time_sec=0,
                        operations_per_sec=0,
                        memory_mb=0,
                        timestamp=self.timestamp,
                        error_message=f"operation={scenario['operation']};error={error_msg}",
                        success=False
                    )
                    self.results.append(result)
    
    async def benchmark_single_get(self, checkpointer, channels: int, checkpoints: int = 1) -> float:
        """Benchmark single checkpoint GET operation."""
        # Generate test data
        checkpoint, metadata, channel_versions = self.generate_checkpoint(channels, 10)
        
        config = {
            "configurable": {
                "thread_id": f"single_get_test_{int(time.time()*1000)}",
                "checkpoint_ns": "",
                "checkpoint_id": checkpoint["id"]
            }
        }
        
        # Store checkpoint first
        if hasattr(checkpointer, 'aput'):
            await checkpointer.aput(config, checkpoint, metadata, channel_versions)
        else:
            checkpointer.put(config, checkpoint, metadata, channel_versions)
        
        # Measure GET operation
        start_time = time.time()
        
        if hasattr(checkpointer, 'aget_tuple'):
            await checkpointer.aget_tuple(config)
        else:
            checkpointer.get_tuple(config)
        
        end_time = time.time()
        
        return (end_time - start_time) * 1000  # Convert to ms
    
    async def benchmark_channel_loading(self, checkpointer, channels: int, checkpoints: int = 1) -> float:
        """Benchmark channel value loading operation."""
        checkpoint, metadata, channel_versions = self.generate_checkpoint(channels, 10)
        
        thread_id = f"channel_test_{int(time.time()*1000)}"
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": checkpoint["id"]
            }
        }
        
        # Store checkpoint first
        if hasattr(checkpointer, 'aput'):
            await checkpointer.aput(config, checkpoint, metadata, channel_versions)
        else:
            checkpointer.put(config, checkpoint, metadata, channel_versions)
        
        # Measure channel loading (use get_tuple as proxy)
        start_time = time.time()
        
        if hasattr(checkpointer, 'aget_tuple'):
            await checkpointer.aget_tuple(config)
        else:
            checkpointer.get_tuple(config)
        
        end_time = time.time()
        
        return (end_time - start_time) * 1000  # Convert to ms
    
    async def benchmark_list_operations(self, checkpointer, channels: int, checkpoints: int) -> float:
        """Benchmark list operations."""
        # Setup multiple checkpoints
        for i in range(checkpoints):
            checkpoint, metadata, channel_versions = self.generate_checkpoint(channels, 1)
            thread_id = f"list_test_{i}_{int(time.time()*1000)}"
            
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "",
                    "checkpoint_id": checkpoint["id"]
                }
            }
            
            if hasattr(checkpointer, 'aput'):
                await checkpointer.aput(config, checkpoint, metadata, channel_versions)
            else:
                checkpointer.put(config, checkpoint, metadata, channel_versions)
        
        # Measure list operation
        start_time = time.time()
        
        if hasattr(checkpointer, 'alist'):
            results = []
            async for item in checkpointer.alist(None, limit=checkpoints):
                results.append(item)
        elif hasattr(checkpointer, 'list'):
            results = list(checkpointer.list(None, limit=checkpoints))
        
        end_time = time.time()
        
        return (end_time - start_time) * 1000  # Convert to ms
    
    def generate_checkpoint(self, channels: int, size_kb: int = 10) -> tuple:
        """Generate checkpoint with specified channel count and payload size."""
        channel_values = {}
        channel_versions = {}
        
        # Generate data to reach target size
        data_per_channel = "x" * (size_kb * 1024 // channels) if channels > 0 else "x" * (size_kb * 1024)
        
        for i in range(channels):
            channel_name = f"channel_{i}"
            channel_values[channel_name] = f"value_{i}_{data_per_channel}"
            channel_versions[channel_name] = i + 1
        
        # Use string-based checkpoint ID to avoid Redis Tag operator issues
        checkpoint_id = f"checkpoint_{int(time.time() * 1000000)}"
        
        checkpoint = {
            "v": 1,
            "id": checkpoint_id,
            "ts": "2024-01-01T00:00:00+00:00",
            "channel_versions": channel_versions,
            "channel_values": channel_values,
            "versions_seen": {},
            "pending_sends": []
        }
        
        metadata = {"source": "benchmark", "step": 1, "writes": {}}
        
        return checkpoint, metadata, channel_versions
    
    def save_individual_reports(self):
        """Save individual JSON files for each implementation."""
        print(f"\n💾 Saving individual implementation reports...")
        
        # Group by implementation
        by_impl = {}
        for result in self.results:
            impl = result.implementation
            if impl not in by_impl:
                by_impl[impl] = []
            by_impl[impl].append(result)
        
        # Save individual files
        for impl, results in by_impl.items():
            version = results[0].version if results else "unknown"
            filename = f"{impl}-{version}.json"
            filepath = self.reports_dir / filename
            
            report_data = {
                "implementation": impl,
                "version": version,
                "timestamp": self.timestamp,
                "total_tests": len(results),
                "scales_tested": sorted(set(r.scale for r in results if r.success)),
                "successful_tests": sum(1 for r in results if r.success),
                "failed_tests": sum(1 for r in results if not r.success),
                "results": [asdict(r) for r in results]
            }
            
            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            print(f"  📄 {filename}")
    
    def save_summary_report(self):
        """Save consolidated summary."""
        summary_data = {
            "timestamp": self.timestamp,
            "benchmark_type": "fanout_to_subgraph",
            "scales_tested": self.scales_to_test,
            "implementations_attempted": list(self.checkpointers.keys()),
            "total_tests": len(self.results),
            "successful_tests": sum(1 for r in self.results if r.success),
            "failed_tests": sum(1 for r in self.results if not r.success),
            "import_failures": failed_imports,
            "results": [asdict(r) for r in self.results]
        }
        
        summary_path = self.reports_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"  📊 summary.json")
    
    def create_performance_plots(self):
        """Generate comprehensive performance plots for all operation types."""
        if not PLOTTING_AVAILABLE:
            print("⚠️  Skipping plots - matplotlib not available")
            return
        
        print(f"\n📊 Generating comprehensive performance plots...")
        
        # Prepare data
        successful_results = [r for r in self.results if r.success]
        if not successful_results:
            print("❌ No successful results to plot")
            return
        
        # 1. Fanout scaling plots (original functionality)
        self.create_fanout_scaling_plots(successful_results)
        
        # 2. Individual operation comparison plots
        self.create_individual_operation_plots(successful_results)
        
        # 3. Comprehensive comparison dashboard
        self.create_comparison_dashboard(successful_results)
    
    def create_fanout_scaling_plots(self, successful_results):
        """Create fanout-to-subgraph scaling plots."""
        # Filter fanout results (those without operation metadata)
        fanout_results = [r for r in successful_results if not r.error_message or 'operation=' not in r.error_message]
        
        if not fanout_results:
            print("⚠️  No fanout scaling results to plot")
            return
        
        # Group by implementation
        by_impl = {}
        for result in fanout_results:
            impl = result.implementation
            if impl not in by_impl:
                by_impl[impl] = {'scales': [], 'times': [], 'ops_per_sec': []}
            by_impl[impl]['scales'].append(result.scale)
            by_impl[impl]['times'].append(result.execution_time_sec)
            by_impl[impl]['ops_per_sec'].append(result.operations_per_sec)
        
        if not by_impl:
            return
        
        # Create execution time plot
        plt.figure(figsize=(12, 8))
        for impl, data in by_impl.items():
            if data['scales']:  # Only plot if we have data
                sorted_data = sorted(zip(data['scales'], data['times']))
                scales, times = zip(*sorted_data)
                plt.plot(scales, times, 'o-', label=impl.capitalize(), linewidth=2, markersize=8)
        
        plt.xlabel('Parallel Graph Executions', fontsize=12)
        plt.ylabel('Time (s)', fontsize=12) 
        plt.title('Fanout-to-Subgraph Scaling Performance', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = self.reports_dir / "fanout_scaling_time.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  📈 fanout_scaling_time.png")
        
        # Create throughput plot
        plt.figure(figsize=(12, 8))
        for impl, data in by_impl.items():
            if data['scales']:  # Only plot if we have data
                sorted_data = sorted(zip(data['scales'], data['ops_per_sec']))
                scales, ops_per_sec = zip(*sorted_data)
                plt.plot(scales, ops_per_sec, 'o-', label=impl.capitalize(), linewidth=2, markersize=8)
        
        plt.xlabel('Parallel Graph Executions', fontsize=12)
        plt.ylabel('Operations per Second', fontsize=12)
        plt.title('Fanout-to-Subgraph Throughput', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        throughput_path = self.reports_dir / "fanout_scaling_throughput.png"
        plt.savefig(throughput_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  📈 fanout_scaling_throughput.png")
    
    def create_individual_operation_plots(self, successful_results):
        """Create plots for individual operations (GET, channel loading, list)."""
        # Filter individual operation results
        operation_results = [r for r in successful_results if r.error_message and 'operation=' in r.error_message]
        
        if not operation_results:
            print("⚠️  No individual operation results to plot")
            return
        
        # Parse operation types
        operations = {}
        for result in operation_results:
            # Extract operation type from error_message metadata
            parts = result.error_message.split(';')
            operation = None
            for part in parts:
                if part.startswith('operation='):
                    operation = part.split('=')[1]
                    break
            
            if operation:
                if operation not in operations:
                    operations[operation] = {}
                if result.implementation not in operations[operation]:
                    operations[operation][result.implementation] = []
                operations[operation][result.implementation].append(result.execution_time_sec * 1000)  # Convert to ms
        
        # Create bar charts for each operation
        for operation, impl_data in operations.items():
            plt.figure(figsize=(10, 6))
            
            implementations = list(impl_data.keys())
            avg_times = [statistics.mean(times) for times in impl_data.values()]
            
            bars = plt.bar(implementations, avg_times)
            
            # Color bars differently for Redis variants
            for i, impl in enumerate(implementations):
                if 'redis' in impl:
                    bars[i].set_color('red' if impl == 'redis' else 'darkred')
                elif impl == 'memory':
                    bars[i].set_color('lightgray')
            
            plt.title(f'{operation.replace("_", " ").title()} Performance', fontsize=14)
            plt.xlabel('Implementation', fontsize=12)
            plt.ylabel('Average Time (ms)', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            plot_path = self.reports_dir / f"operation_{operation}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  📈 operation_{operation}.png")
    
    def create_comparison_dashboard(self, successful_results):
        """Create a comprehensive comparison dashboard."""
        # Create a 2x2 subplot dashboard
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Fanout scaling comparison (top-left)
        fanout_results = [r for r in successful_results if not r.error_message or 'operation=' not in r.error_message]
        if fanout_results:
            by_impl = {}
            for result in fanout_results:
                impl = result.implementation
                if impl not in by_impl:
                    by_impl[impl] = {'scales': [], 'ops_per_sec': []}
                by_impl[impl]['scales'].append(result.scale)
                by_impl[impl]['ops_per_sec'].append(result.operations_per_sec)
            
            for impl, data in by_impl.items():
                if data['scales']:
                    sorted_data = sorted(zip(data['scales'], data['ops_per_sec']))
                    scales, ops_per_sec = zip(*sorted_data)
                    ax1.plot(scales, ops_per_sec, 'o-', label=impl.capitalize(), linewidth=2)
            
            ax1.set_title('Fanout Scaling Throughput', fontsize=12)
            ax1.set_xlabel('Parallel Executions')
            ax1.set_ylabel('Ops/sec')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Individual operations comparison (top-right)
        operation_results = [r for r in successful_results if r.error_message and 'operation=' in r.error_message]
        if operation_results:
            operations_summary = {}
            for result in operation_results:
                parts = result.error_message.split(';')
                operation = None
                for part in parts:
                    if part.startswith('operation='):
                        operation = part.split('=')[1].replace('_', ' ').title()
                        break
                
                if operation:
                    if result.implementation not in operations_summary:
                        operations_summary[result.implementation] = {}
                    if operation not in operations_summary[result.implementation]:
                        operations_summary[result.implementation][operation] = []
                    operations_summary[result.implementation][operation].append(result.execution_time_sec * 1000)
            
            # Create grouped bar chart
            if operations_summary:
                implementations = list(operations_summary.keys())
                all_operations = set()
                for impl_ops in operations_summary.values():
                    all_operations.update(impl_ops.keys())
                all_operations = sorted(list(all_operations))
                
                x = range(len(implementations))
                width = 0.25
                
                for i, operation in enumerate(all_operations[:3]):  # Limit to 3 operations
                    values = []
                    for impl in implementations:
                        if operation in operations_summary[impl]:
                            values.append(statistics.mean(operations_summary[impl][operation]))
                        else:
                            values.append(0)
                    
                    offset = (i - 1) * width
                    ax2.bar([xi + offset for xi in x], values, width, label=operation)
                
                ax2.set_title('Individual Operations Performance', fontsize=12)
                ax2.set_xlabel('Implementation')
                ax2.set_ylabel('Avg Time (ms)')
                ax2.set_xticks(x)
                ax2.set_xticklabels(implementations, rotation=45)
                ax2.legend()
                ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Performance comparison at scale 900 (bottom-left)
        scale_900_results = [r for r in fanout_results if r.scale == 900]
        if scale_900_results:
            implementations = [r.implementation for r in scale_900_results]
            times = [r.execution_time_sec for r in scale_900_results]
            
            bars = ax3.bar(implementations, times)
            for i, impl in enumerate(implementations):
                if impl == 'redis':
                    bars[i].set_color('red')
                elif impl == 'memory':
                    bars[i].set_color('lightgray')
            
            ax3.set_title('Performance at Scale 900', fontsize=12)
            ax3.set_xlabel('Implementation')
            ax3.set_ylabel('Execution Time (s)')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Redis vs Others comparison (bottom-right)
        if fanout_results:
            redis_results = [r for r in fanout_results if r.implementation == 'redis']
            other_results = [r for r in fanout_results if r.implementation != 'redis' and r.implementation != 'memory']
            
            if redis_results and other_results:
                redis_avg = statistics.mean([r.execution_time_sec for r in redis_results])
                
                other_avgs = {}
                for result in other_results:
                    if result.implementation not in other_avgs:
                        other_avgs[result.implementation] = []
                    other_avgs[result.implementation].append(result.execution_time_sec)
                
                implementations = ['Redis'] + [impl.capitalize() for impl in other_avgs.keys()]
                times = [redis_avg] + [statistics.mean(times) for times in other_avgs.values()]
                
                bars = ax4.bar(implementations, times)
                bars[0].set_color('red')  # Redis bar in red
                
                ax4.set_title('Redis vs Other Persistent Stores', fontsize=12)
                ax4.set_xlabel('Implementation')
                ax4.set_ylabel('Avg Execution Time (s)')
                ax4.tick_params(axis='x', rotation=45)
                ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        dashboard_path = self.reports_dir / "performance_dashboard.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  📊 performance_dashboard.png")
    
    def print_summary(self):
        """Print summary of results."""
        print(f"\n" + "="*80)
        print(f"🎯 BENCHMARK RESULTS")
        print(f"="*80)
        
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        print(f"✅ Successful tests: {len(successful_results)}")
        print(f"❌ Failed tests: {len(failed_results)}")
        
        if failed_results:
            print(f"\n❌ FAILURES:")
            for result in failed_results:
                print(f"   {result.implementation} at scale {result.scale}: {result.error_message}")
        
        if successful_results:
            print(f"\n📊 PERFORMANCE AT SCALE 900:")
            scale_900_results = [r for r in successful_results if r.scale == 900]
            if scale_900_results:
                sorted_900 = sorted(scale_900_results, key=lambda r: r.execution_time_sec)
                for i, result in enumerate(sorted_900, 1):
                    print(f"   {i}. {result.implementation}: {result.execution_time_sec:.2f}s ({result.operations_per_sec:.0f} ops/sec)")
            else:
                print("   No implementations successfully tested at scale 900")
                
                # Show highest scale that worked
                max_scale = max(r.scale for r in successful_results)
                print(f"\n📊 PERFORMANCE AT HIGHEST WORKING SCALE ({max_scale}):")
                max_scale_results = [r for r in successful_results if r.scale == max_scale]
                sorted_max = sorted(max_scale_results, key=lambda r: r.execution_time_sec)
                for i, result in enumerate(sorted_max, 1):
                    print(f"   {i}. {result.implementation}: {result.execution_time_sec:.2f}s ({result.operations_per_sec:.0f} ops/sec)")
        
        print(f"\n📁 All results saved to: {self.reports_dir}")
    
    async def cleanup(self):
        """Clean up resources."""
        print(f"\n🧹 Cleaning up...")
        
        # Close async contexts
        context_names = [
            'redis_context', 'redis_shallow_context', 
            'sqlite_context', 'sqlite_shallow_context',
            'mongodb_context', 'mongodb_shallow_context',
            'mysql_context', 'mysql_shallow_context'
        ]
        
        for context_name in context_names:
            if hasattr(self, context_name):
                context = getattr(self, context_name)
                try:
                    await context.__aexit__(None, None, None)
                    print(f"  ✅ Closed {context_name}")
                except Exception as e:
                    print(f"  ⚠️  Error closing {context_name}: {e}")
        
        # Stop containers
        for name, container in self.containers.items():
            try:
                container.stop()
                print(f"  ✅ Stopped {name} container")
            except Exception as e:
                print(f"  ⚠️  Error stopping {name}: {e}")


async def main():
    """Run the checkpointer benchmark."""
    print("🚀 Starting Checkpointer Benchmark")
    print("="*80)
    
    benchmark = CheckpointerBenchmark()
    
    try:
        await benchmark.setup_checkpointers()
        await benchmark.run_benchmark()
        
        benchmark.save_individual_reports()
        benchmark.save_summary_report()
        benchmark.create_performance_plots()
        benchmark.print_summary()
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await benchmark.cleanup()


if __name__ == "__main__":
    asyncio.run(main())