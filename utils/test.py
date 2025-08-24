"""
performance_test.py - Measure system performance metrics
Run this after starting your app to get real performance numbers
"""

import time
import psutil
import os
import sys
import json
import numpy as np
import pandas as pd
from typing import List, Dict
import tracemalloc
import gc

# Add your project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import boot
from rag.search.config import SearchMode
from db.manager import DatabaseManager
from config import SQLITE_DB_PATH, SQLITE_TABLE_NAME

def measure_startup_time():
    """Measure application initialization time."""
    print("\n" + "="*60)
    print("MEASURING STARTUP PERFORMANCE")
    print("="*60)
    
    # Measure cold start
    gc.collect()
    start = time.time()
    app, retriever = boot()
    startup_time = time.time() - start
    
    print(f"✓ Startup time: {startup_time:.3f} seconds")
    return startup_time, retriever

def measure_memory_usage():
    """Measure current memory usage."""
    print("\n" + "="*60)
    print("MEASURING MEMORY USAGE")
    print("="*60)
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    memory_stats = {
        'rss_mb': memory_info.rss / 1024 / 1024,
        'vms_mb': memory_info.vms / 1024 / 1024,
        'percent': process.memory_percent()
    }
    
    print(f"✓ RSS Memory: {memory_stats['rss_mb']:.1f} MB")
    print(f"✓ Virtual Memory: {memory_stats['vms_mb']:.1f} MB")
    print(f"✓ Memory Percent: {memory_stats['percent']:.2f}%")
    
    return memory_stats

def measure_search_latency(retriever, num_queries=50):
    """Measure search latency with various query types."""
    print("\n" + "="*60)
    print(f"MEASURING SEARCH LATENCY ({num_queries} queries)")
    print("="*60)
    
    # Test queries of different types
    test_queries = [
        # Short queries
        "IBAN", "BIC", "currency", "amount", "date",
        # Medium queries  
        "validate IBAN", "check currency", "payment amount",
        "transaction date", "reference number",
        # Long queries
        "How to validate IBAN for SEPA payments",
        "Check if currency code is valid for transaction",
        "Validate payment amount exceeds minimum threshold",
        # Technical queries
        "MT103", "MT202", "pacs.008", "pain.001", "ISO20022",
        # Error-based queries
        "BANSTA error", "validation failed", "invalid format",
        # Empty and edge cases
        "", "a", "123", "!!!", "test test test"
    ]
    
    # Repeat queries to get more samples
    queries = (test_queries * (num_queries // len(test_queries) + 1))[:num_queries]
    
    results = {
        'hybrid': [],
        'keyword': [],
        'semantic': [],
        'fuzzy': []
    }
    
    # Test each search mode
    for mode_name, mode in [
        ('hybrid', SearchMode.HYBRID),
        ('keyword', SearchMode.KEYWORD),
        ('semantic', SearchMode.SEMANTIC),
        ('fuzzy', SearchMode.FUZZY)
    ]:
        print(f"\nTesting {mode_name.upper()} mode...")
        latencies = []
        
        for i, query in enumerate(queries):
            start = time.perf_counter()
            try:
                _ = retriever.search_rules(
                    query=query,
                    mode=mode,
                    top_k=10
                )
                latency = (time.perf_counter() - start) * 1000  # Convert to ms
                latencies.append(latency)
            except Exception as e:
                print(f"  Error on query '{query[:20]}...': {e}")
                latencies.append(None)
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{num_queries} queries")
        
        # Calculate statistics
        valid_latencies = [l for l in latencies if l is not None]
        if valid_latencies:
            stats = {
                'mean': np.mean(valid_latencies),
                'median': np.median(valid_latencies),
                'p95': np.percentile(valid_latencies, 95),
                'p99': np.percentile(valid_latencies, 99),
                'min': np.min(valid_latencies),
                'max': np.max(valid_latencies),
                'std': np.std(valid_latencies)
            }
            results[mode_name] = stats
            
            print(f"  Mean: {stats['mean']:.2f} ms")
            print(f"  P50:  {stats['median']:.2f} ms")
            print(f"  P95:  {stats['p95']:.2f} ms")
            print(f"  P99:  {stats['p99']:.2f} ms")
    
    return results

def measure_filter_performance(retriever):
    """Measure performance with filters applied."""
    print("\n" + "="*60)
    print("MEASURING FILTER PERFORMANCE")
    print("="*60)
    
    filter_options = retriever.filter_options
    
    test_cases = [
        {'name': 'No filters', 'filters': {}},
        {'name': 'Single filter', 'filters': {
            'rule_type': [filter_options['rule_type'][0]] if filter_options['rule_type'] else []
        }},
        {'name': 'Multiple filters', 'filters': {
            'rule_type': filter_options['rule_type'][:2] if len(filter_options['rule_type']) > 1 else [],
            'country': filter_options['country'][:2] if len(filter_options['country']) > 1 else []
        }}
    ]
    
    results = []
    for test in test_cases:
        print(f"\nTesting: {test['name']}")
        latencies = []
        
        for _ in range(20):  # 20 queries per test case
            start = time.perf_counter()
            _ = retriever.search_rules(
                query="payment validation",
                **test['filters'],
                mode=SearchMode.HYBRID,
                top_k=10
            )
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)
        
        stats = {
            'test': test['name'],
            'mean_ms': np.mean(latencies),
            'p95_ms': np.percentile(latencies, 95)
        }
        results.append(stats)
        print(f"  Mean: {stats['mean_ms']:.2f} ms, P95: {stats['p95_ms']:.2f} ms")
    
    return results

def measure_database_performance():
    """Measure database query performance."""
    print("\n" + "="*60)
    print("MEASURING DATABASE PERFORMANCE")
    print("="*60)
    
    db_manager = DatabaseManager(SQLITE_DB_PATH, SQLITE_TABLE_NAME)
    
    # Measure read performance
    read_times = []
    for _ in range(10):
        start = time.perf_counter()
        rules = db_manager.get_rules()
        read_time = (time.perf_counter() - start) * 1000
        read_times.append(read_time)
    
    print(f"✓ Database read (all rules):")
    print(f"  Mean: {np.mean(read_times):.2f} ms")
    print(f"  P95:  {np.percentile(read_times, 95):.2f} ms")
    print(f"  Rules loaded: {len(rules)}")
    
    return {
        'read_mean_ms': np.mean(read_times),
        'read_p95_ms': np.percentile(read_times, 95),
        'num_rules': len(rules)
    }

def measure_index_building(retriever):
    """Measure time to build indices."""
    print("\n" + "="*60)
    print("MEASURING INDEX BUILD TIME")
    print("="*60)
    
    # Force rebuild of indices
    start = time.perf_counter()
    retriever._build_indices()
    build_time = (time.perf_counter() - start) * 1000
    
    print(f"✓ Total index build time: {build_time:.2f} ms")
    print(f"  Rules indexed: {len(retriever.rules)}")
    
    return build_time

def generate_performance_report(all_results):
    """Generate a formatted performance report."""
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY REPORT")
    print("="*60)
    
    report = f"""
## System Performance Metrics

### Startup Performance
- **Initialization Time**: {all_results['startup']:.2f} seconds
- **Memory at Startup**: {all_results['memory']['rss_mb']:.1f} MB

### Search Latency (milliseconds)
| Mode     | Mean  | P50   | P95   | P99   |
|----------|-------|-------|-------|-------|
| Hybrid   | {all_results['search']['hybrid']['mean']:.1f} | {all_results['search']['hybrid']['median']:.1f} | {all_results['search']['hybrid']['p95']:.1f} | {all_results['search']['hybrid']['p99']:.1f} |
| Keyword  | {all_results['search']['keyword']['mean']:.1f} | {all_results['search']['keyword']['median']:.1f} | {all_results['search']['keyword']['p95']:.1f} | {all_results['search']['keyword']['p99']:.1f} |
| Semantic | {all_results['search']['semantic']['mean']:.1f} | {all_results['search']['semantic']['median']:.1f} | {all_results['search']['semantic']['p95']:.1f} | {all_results['search']['semantic']['p99']:.1f} |
| Fuzzy    | {all_results['search']['fuzzy']['mean']:.1f} | {all_results['search']['fuzzy']['median']:.1f} | {all_results['search']['fuzzy']['p95']:.1f} | {all_results['search']['fuzzy']['p99']:.1f} |

### Database Performance
- **Full Read Mean**: {all_results['database']['read_mean_ms']:.2f} ms
- **Full Read P95**: {all_results['database']['read_p95_ms']:.2f} ms
- **Total Rules**: {all_results['database']['num_rules']}

### Index Building
- **Build Time**: {all_results['index_build']:.1f} ms

### Resource Usage
- **Memory (RSS)**: {all_results['memory']['rss_mb']:.1f} MB
- **Memory (VMS)**: {all_results['memory']['vms_mb']:.1f} MB
- **Memory %**: {all_results['memory']['percent']:.2f}%
"""
    
    print(report)
    
    # Save to file
    with open('performance_report.md', 'w') as f:
        f.write(report)
    print("\n✓ Report saved to performance_report.md")
    
    # Also save raw data as JSON
    with open('performance_data.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print("✓ Raw data saved to performance_data.json")
    
    return report

def main():
    """Run all performance tests."""
    print("\n" + "#"*60)
    print("# VALIDATION RULE SEARCH - PERFORMANCE TESTING")
    print("#"*60)
    
    all_results = {}
    
    try:
        # 1. Startup performance
        startup_time, retriever = measure_startup_time()
        all_results['startup'] = startup_time
        
        # 2. Memory usage
        memory_stats = measure_memory_usage()
        all_results['memory'] = memory_stats
        
        # 3. Search latency
        search_results = measure_search_latency(retriever, num_queries=50)
        all_results['search'] = search_results
        
        # 4. Filter performance
        filter_results = measure_filter_performance(retriever)
        all_results['filters'] = filter_results
        
        # 5. Database performance
        db_results = measure_database_performance()
        all_results['database'] = db_results
        
        # 6. Index building
        index_time = measure_index_building(retriever)
        all_results['index_build'] = index_time
        
        # Generate report
        generate_performance_report(all_results)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        print(traceback.format_exc())
        sys.exit(1)
    
    print("\n" + "#"*60)
    print("# TESTING COMPLETE")
    print("#"*60)

if __name__ == "__main__":
    main()