"""
test_performance.py - Performance tests using pytest
Run with: pytest test_performance.py -v
"""

import time
import psutil
import os
import pytest
import numpy as np
from typing import Dict, List

from app import boot
from rag.search.config import SearchMode
from db.manager import DatabaseManager
from config import SQLITE_DB_PATH, SQLITE_TABLE_NAME


class TestPerformance:
    """Performance test suite for the validation rule search system."""
    
    @classmethod
    def setup_class(cls):
        """Setup that runs once for all tests in this class."""
        print("\nInitializing system for performance tests...")
        cls.app, cls.retriever = boot()
        cls.db_manager = DatabaseManager(SQLITE_DB_PATH, SQLITE_TABLE_NAME)
    
    def test_startup_time(self):
        """Test that system startup is under 2 seconds."""
        start = time.time()
        app, retriever = boot()
        startup_time = time.time() - start
        
        assert startup_time < 2.0, f"Startup took {startup_time:.2f}s, expected < 2s"
        print(f"\n✓ Startup time: {startup_time:.3f} seconds")
    
    def test_memory_usage(self):
        """Test that memory usage is under 1GB."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        assert memory_mb < 1024, f"Memory usage {memory_mb:.1f}MB exceeds 1GB limit"
        print(f"\n✓ Memory usage: {memory_mb:.1f} MB")
    
    @pytest.mark.parametrize("mode,expected_p95", [
        (SearchMode.KEYWORD, 10),      # Keyword should be under 10ms P95
        (SearchMode.SEMANTIC, 20),     # Semantic should be under 20ms P95
        (SearchMode.FUZZY, 100),       # Fuzzy should be under 100ms P95
        (SearchMode.HYBRID, 1000),     # Hybrid should be under 1000ms P95
    ])
    def test_search_latency(self, mode, expected_p95):
        """Test search latency for different modes."""
        test_queries = [
            "IBAN", "BIC", "currency", "amount", "date",
            "validate IBAN", "check currency", "payment amount",
            "MT103", "MT202", "pacs.008", "pain.001", "ISO20022",
        ]
        
        latencies = []
        for query in test_queries:
            start = time.perf_counter()
            results = self.retriever.search_rules(
                query=query,
                mode=mode,
                top_k=10
            )
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)
        
        p95 = np.percentile(latencies, 95)
        mean = np.mean(latencies)
        
        assert p95 < expected_p95, (
            f"{mode.value} mode P95 latency {p95:.1f}ms exceeds {expected_p95}ms limit"
        )
        print(f"\n✓ {mode.value}: Mean={mean:.1f}ms, P95={p95:.1f}ms")
    
    def test_database_read_performance(self):
        """Test database read performance."""
        read_times = []
        for _ in range(10):
            start = time.perf_counter()
            rules = self.db_manager.get_rules()
            read_time = (time.perf_counter() - start) * 1000
            read_times.append(read_time)
        
        mean_time = np.mean(read_times)
        p95_time = np.percentile(read_times, 95)
        
        assert mean_time < 100, f"DB read mean {mean_time:.1f}ms exceeds 100ms limit"
        assert len(rules) == 1157, f"Expected 1157 rules, got {len(rules)}"
        
        print(f"\n✓ DB Read: Mean={mean_time:.1f}ms, P95={p95_time:.1f}ms")
    
    def test_index_build_time(self):
        """Test that index building is under 500ms."""
        start = time.perf_counter()
        self.retriever._build_indices()
        build_time = (time.perf_counter() - start) * 1000
        
        assert build_time < 500, f"Index build {build_time:.1f}ms exceeds 500ms limit"
        print(f"\n✓ Index build time: {build_time:.1f}ms")
    
    def test_filter_performance(self):
        """Test that filters improve performance."""
        # No filters
        start = time.perf_counter()
        results_no_filter = self.retriever.search_rules(
            query="payment",
            mode=SearchMode.HYBRID,
            top_k=10
        )
        time_no_filter = (time.perf_counter() - start) * 1000
        
        # With filter
        filter_options = self.retriever.filter_options
        if filter_options.get('rule_type'):
            start = time.perf_counter()
            results_filtered = self.retriever.search_rules(
                query="payment",
                rule_type=[filter_options['rule_type'][0]],
                mode=SearchMode.HYBRID,
                top_k=10
            )
            time_filtered = (time.perf_counter() - start) * 1000
            
            # Filtered search should be faster or similar
            assert time_filtered <= time_no_filter * 1.5, (
                f"Filtered search slower: {time_filtered:.1f}ms vs {time_no_filter:.1f}ms"
            )
            print(f"\n✓ Filter speedup: {time_no_filter:.1f}ms → {time_filtered:.1f}ms")
    
    def test_concurrent_searches(self):
        """Test system under concurrent load."""
        import concurrent.futures
        
        def search_task(query):
            start = time.perf_counter()
            self.retriever.search_rules(query=query, mode=SearchMode.HYBRID, top_k=5)
            return (time.perf_counter() - start) * 1000
        
        queries = ["IBAN", "BIC", "currency", "payment", "validation"] * 2
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            latencies = list(executor.map(search_task, queries))
        
        p95 = np.percentile(latencies, 95)
        assert p95 < 2000, f"Concurrent P95 {p95:.1f}ms exceeds 2000ms limit"
        print(f"\n✓ Concurrent searches P95: {p95:.1f}ms")


class TestSearchQuality:
    """Test search result quality and correctness."""
    
    @classmethod
    def setup_class(cls):
        """Setup that runs once for all tests in this class."""
        _, cls.retriever = boot()
    
    def test_empty_query_handling(self):
        """Test that empty queries are handled gracefully."""
        results = self.retriever.search_rules(query="", mode=SearchMode.HYBRID, top_k=10)
        assert isinstance(results, list), "Empty query should return a list"
        assert len(results) <= 10, "Should respect top_k limit"
    
    def test_filter_only_search(self):
        """Test searching with filters but no query."""
        filter_options = self.retriever.filter_options
        if filter_options.get('country'):
            results = self.retriever.search_rules(
                query=None,
                country=[filter_options['country'][0]],
                mode=SearchMode.HYBRID,
                top_k=10
            )
            assert isinstance(results, list), "Filter-only search should return results"
            assert all(
                filter_options['country'][0] in r.get('country', []) 
                for r in results
            ), "All results should match the filter"
    
    def test_search_modes_return_results(self):
        """Test that all search modes return valid results."""
        query = "payment validation"
        
        for mode in [SearchMode.HYBRID, SearchMode.KEYWORD, SearchMode.SEMANTIC, SearchMode.FUZZY]:
            results = self.retriever.search_rules(query=query, mode=mode, top_k=5)
            assert isinstance(results, list), f"{mode} should return a list"
            assert len(results) <= 5, f"{mode} should respect top_k"
            assert all('search_score' in r for r in results), f"{mode} results should have scores"


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])

