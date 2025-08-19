# corpus_analysis.ipynb
# Comprehensive Corpus Analysis for Kotlin Validation Rules

# %% [markdown]
# # Corpus Analysis: Kotlin Validation Rules
# 
# This notebook performs comprehensive analysis of the standardized CSV corpus containing
# Kotlin code validation rules for the eBridge-EU system. We examine field completeness,
# data quality, distributions, and embedding characteristics.

# %% Import Libraries
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Configure visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

# %% [markdown]
# ## 1. Data Loading and Initial Inspection

# %% Load Data
def load_corpus(filepath: str) -> pd.DataFrame:
    """Load the corpus with proper type handling."""
    df = pd.read_csv(filepath, encoding='utf-8')
    
    # Parse dates
    date_cols = ['created_at', 'updated_at']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Parse embedding JSON strings
    if 'embedding' in df.columns:
        df['embedding_parsed'] = df['embedding'].apply(parse_embedding)
    
    return df

def parse_embedding(embedding_str: str) -> np.ndarray:
    """Parse embedding JSON string to numpy array."""
    if pd.isna(embedding_str):
        return None
    try:
        emb = json.loads(embedding_str)
        return np.array(emb, dtype=np.float32)
    except:
        return None

# Load the corpus
df = load_corpus('rules_corpus.csv')

print("=" * 80)
print("CORPUS OVERVIEW")
print("=" * 80)
print(f"Total Rules: {len(df):,}")
print(f"Total Columns: {len(df.columns)}")
print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"Date Range: {df['created_at'].min()} to {df['updated_at'].max()}")
print("\nColumn Names:")
for col in df.columns:
    print(f"  - {col}")

# %% [markdown]
# ## 2. Field Completeness Analysis

# %% Field Completeness
def analyze_completeness(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze field completeness across the corpus."""
    completeness = []
    
    for col in df.columns:
        if col == 'embedding_parsed':
            continue
            
        total = len(df)
        non_null = df[col].notna().sum()
        non_empty = 0
        
        if df[col].dtype == 'object':
            # For string fields, check for empty strings
            non_empty = ((df[col].notna()) & (df[col].str.strip() != '')).sum()
        else:
            non_empty = non_null
            
        completeness.append({
            'Field': col,
            'Non-Null': non_null,
            'Non-Null %': 100 * non_null / total,
            'Non-Empty': non_empty,
            'Non-Empty %': 100 * non_empty / total,
            'Data Type': str(df[col].dtype)
        })
    
    return pd.DataFrame(completeness).sort_values('Non-Empty %', ascending=False)

completeness_df = analyze_completeness(df)

print("\n" + "=" * 80)
print("FIELD COMPLETENESS ANALYSIS")
print("=" * 80)
print(completeness_df.to_string(index=False))

# Visualize completeness
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Critical fields for retrieval
critical_fields = ['rule_id', 'rule_name', 'keywords', 'embedding', 
                  'llm_description', 'bansta_error_code', 'iso_error_code']
critical_completeness = completeness_df[completeness_df['Field'].isin(critical_fields)]

ax1.barh(critical_completeness['Field'], critical_completeness['Non-Empty %'])
ax1.set_xlabel('Completeness (%)')
ax1.set_title('Critical Field Completeness')
ax1.axvline(x=95, color='g', linestyle='--', alpha=0.5, label='95% Target')
ax1.axvline(x=100, color='b', linestyle='-', alpha=0.3)
ax1.legend()
ax1.grid(axis='x', alpha=0.3)

# Completeness by category
categories = {
    'Identifiers': ['rule_id', 'rule_name'],
    'Descriptions': ['rule_description', 'description_en', 'description_de', 'llm_description'],
    'Error Codes': ['bansta_error_code', 'iso_error_code'],
    'Search Fields': ['keywords', 'embedding'],
    'Tags': ['rule_type', 'country', 'business_type', 'party_agent'],
    'Metadata': ['relevance', 'version_major', 'version_minor', 'created_at', 'updated_at'],
    'Implementation': ['rule_code']
}

category_completeness = []
for cat, fields in categories.items():
    cat_fields = completeness_df[completeness_df['Field'].isin(fields)]
    if not cat_fields.empty:
        category_completeness.append({
            'Category': cat,
            'Avg Completeness': cat_fields['Non-Empty %'].mean()
        })

cat_df = pd.DataFrame(category_completeness)
ax2.bar(cat_df['Category'], cat_df['Avg Completeness'])
ax2.set_ylabel('Average Completeness (%)')
ax2.set_title('Completeness by Field Category')
ax2.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90% Target')
ax2.tick_params(axis='x', rotation=45)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('field_completeness.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 3. Categorical Field Distributions (Tags)

# %% Tag Distributions
def analyze_categorical_field(df: pd.DataFrame, field: str) -> Dict[str, Any]:
    """Analyze distribution of categorical fields."""
    if field not in df.columns:
        return None
    
    # Get non-null values
    values = df[field].dropna()
    values = values[values.astype(str).str.strip() != '']
    
    value_counts = values.value_counts()
    
    return {
        'field': field,
        'unique_values': len(value_counts),
        'total_assignments': len(values),
        'coverage': 100 * len(values) / len(df),
        'top_values': value_counts.head(10).to_dict(),
        'distribution': value_counts,
        'entropy': -sum((c/len(values)) * np.log2(c/len(values)) for c in value_counts.values)
    }

# Analyze tag fields
tag_fields = ['rule_type', 'country', 'business_type', 'party_agent']
tag_analysis = {}

print("\n" + "=" * 80)
print("CATEGORICAL TAG ANALYSIS")
print("=" * 80)

for field in tag_fields:
    analysis = analyze_categorical_field(df, field)
    if analysis:
        tag_analysis[field] = analysis
        print(f"\n{field.upper().replace('_', ' ')}:")
        print(f"  Coverage: {analysis['coverage']:.1f}%")
        print(f"  Unique values: {analysis['unique_values']}")
        print(f"  Entropy: {analysis['entropy']:.2f} bits")
        print(f"  Top 5 values:")
        for val, count in list(analysis['top_values'].items())[:5]:
            print(f"    - {val}: {count} ({100*count/analysis['total_assignments']:.1f}%)")

# Visualize tag distributions
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, (field, analysis) in enumerate(tag_analysis.items()):
    if idx < 4:
        ax = axes[idx]
        top_10 = analysis['distribution'].head(10)
        
        # Create bar plot
        bars = ax.bar(range(len(top_10)), top_10.values)
        ax.set_xticks(range(len(top_10)))
        ax.set_xticklabels(top_10.index, rotation=45, ha='right')
        ax.set_title(f'{field.replace("_", " ").title()} Distribution\n' + 
                    f'Coverage: {analysis["coverage"]:.1f}%, Unique: {analysis["unique_values"]}')
        ax.set_ylabel('Count')
        ax.grid(axis='y', alpha=0.3)
        
        # Color bars by frequency
        norm = plt.Normalize(vmin=min(top_10.values), vmax=max(top_10.values))
        colors = plt.cm.viridis(norm(top_10.values))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

plt.tight_layout()
plt.savefig('tag_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 4. Text Field Analysis

# %% Text Field Statistics
def analyze_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze text field characteristics."""
    text_fields = ['rule_name', 'rule_description', 'description_en', 
                   'description_de', 'llm_description', 'keywords']
    
    stats = []
    for field in text_fields:
        if field in df.columns:
            non_empty = df[field].notna() & (df[field].astype(str).str.strip() != '')
            text_data = df.loc[non_empty, field].astype(str)
            
            if len(text_data) > 0:
                lengths = text_data.str.len()
                word_counts = text_data.str.split().str.len()
                
                # Language detection for descriptions
                has_special_chars = text_data.str.contains('[äöüÄÖÜß]', regex=True).sum()
                
                stats.append({
                    'Field': field,
                    'Coverage %': 100 * len(text_data) / len(df),
                    'Avg Length': lengths.mean(),
                    'Median Length': lengths.median(),
                    'Max Length': lengths.max(),
                    'Min Length': lengths.min(),
                    'Avg Words': word_counts.mean(),
                    'Median Words': word_counts.median(),
                    'Max Words': word_counts.max(),
                    'German Chars %': 100 * has_special_chars / len(text_data) if 'de' in field else 0
                })
    
    return pd.DataFrame(stats)

text_stats = analyze_text_fields(df)

print("\n" + "=" * 80)
print("TEXT FIELD STATISTICS")
print("=" * 80)
print(text_stats.to_string(index=False, float_format='%.1f'))

# Visualize text field characteristics
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Word count distributions
ax1 = axes[0, 0]
x = np.arange(len(text_stats))
width = 0.35
ax1.bar(x - width/2, text_stats['Avg Words'], width, label='Average', alpha=0.8)
ax1.bar(x + width/2, text_stats['Median Words'], width, label='Median', alpha=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels(text_stats['Field'], rotation=45, ha='right')
ax1.set_ylabel('Word Count')
ax1.set_title('Text Field Word Counts')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Coverage
ax2 = axes[0, 1]
bars = ax2.barh(text_stats['Field'], text_stats['Coverage %'])
ax2.set_xlabel('Coverage (%)')
ax2.set_title('Text Field Coverage')
ax2.axvline(x=80, color='orange', linestyle='--', alpha=0.5, label='80% Target')
ax2.axvline(x=95, color='green', linestyle='--', alpha=0.5, label='95% Target')
# Color bars by coverage
for i, (bar, cov) in enumerate(zip(bars, text_stats['Coverage %'])):
    if cov >= 95:
        bar.set_color('green')
    elif cov >= 80:
        bar.set_color('orange')
    else:
        bar.set_color('red')
ax2.legend()
ax2.grid(axis='x', alpha=0.3)

# Character length distribution
ax3 = axes[1, 0]
ax3.bar(text_stats['Field'], text_stats['Avg Length'])
ax3.set_ylabel('Average Character Length')
ax3.set_title('Average Text Length by Field')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(axis='y', alpha=0.3)

# Min/Max range
ax4 = axes[1, 1]
fields = text_stats['Field']
mins = text_stats['Min Length']
maxs = text_stats['Max Length']
avgs = text_stats['Avg Length']

x_pos = np.arange(len(fields))
ax4.scatter(x_pos, mins, marker='v', s=100, label='Min', color='blue')
ax4.scatter(x_pos, maxs, marker='^', s=100, label='Max', color='red')
ax4.scatter(x_pos, avgs, marker='o', s=100, label='Avg', color='green')

for i in range(len(fields)):
    ax4.plot([i, i], [mins.iloc[i], maxs.iloc[i]], 'k-', alpha=0.3)

ax4.set_xticks(x_pos)
ax4.set_xticklabels(fields, rotation=45, ha='right')
ax4.set_ylabel('Character Length')
ax4.set_title('Text Length Range by Field')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('text_field_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 5. Keyword Analysis

# %% Keyword Analysis
def analyze_keywords(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze keyword field in detail."""
    keyword_data = df['keywords'].dropna()
    keyword_data = keyword_data[keyword_data.str.strip() != '']
    
    # Parse keywords (comma-separated)
    all_keywords = []
    keyword_counts_per_rule = []
    
    for keywords_str in keyword_data:
        keywords = [k.strip().lower() for k in str(keywords_str).split(',') if k.strip()]
        all_keywords.extend(keywords)
        keyword_counts_per_rule.append(len(keywords))
    
    # Calculate statistics
    unique_keywords = pd.Series(all_keywords).value_counts()
    
    # Analyze keyword patterns
    keyword_lengths = [len(k) for k in all_keywords]
    
    return {
        'total_rules_with_keywords': len(keyword_data),
        'coverage': 100 * len(keyword_data) / len(df),
        'total_unique_keywords': len(unique_keywords),
        'total_keyword_occurrences': len(all_keywords),
        'avg_keywords_per_rule': np.mean(keyword_counts_per_rule),
        'median_keywords_per_rule': np.median(keyword_counts_per_rule),
        'max_keywords_per_rule': np.max(keyword_counts_per_rule),
        'min_keywords_per_rule': np.min(keyword_counts_per_rule),
        'avg_keyword_length': np.mean(keyword_lengths),
        'top_keywords': unique_keywords.head(20).to_dict(),
        'keyword_distribution': keyword_counts_per_rule,
        'singleton_keywords': (unique_keywords == 1).sum(),
        'high_freq_keywords': (unique_keywords >= 10).sum()
    }

keyword_analysis = analyze_keywords(df)

print("\n" + "=" * 80)
print("KEYWORD ANALYSIS")
print("=" * 80)
print(f"Coverage: {keyword_analysis['coverage']:.1f}%")
print(f"Total unique keywords: {keyword_analysis['total_unique_keywords']:,}")
print(f"Total keyword occurrences: {keyword_analysis['total_keyword_occurrences']:,}")
print(f"Keywords per rule: {keyword_analysis['avg_keywords_per_rule']:.1f} avg, "
      f"{keyword_analysis['median_keywords_per_rule']:.0f} median")
print(f"Range: {keyword_analysis['min_keywords_per_rule']:.0f} - "
      f"{keyword_analysis['max_keywords_per_rule']:.0f}")
print(f"Avg keyword length: {keyword_analysis['avg_keyword_length']:.1f} chars")
print(f"Singleton keywords: {keyword_analysis['singleton_keywords']} "
      f"({100*keyword_analysis['singleton_keywords']/keyword_analysis['total_unique_keywords']:.1f}%)")
print(f"High-frequency keywords (≥10): {keyword_analysis['high_freq_keywords']}")

print("\nTop 10 Keywords:")
for keyword, count in list(keyword_analysis['top_keywords'].items())[:10]:
    freq = 100 * count / keyword_analysis['total_rules_with_keywords']
    print(f"  - '{keyword}': {count} occurrences ({freq:.1f}% of rules)")

# Visualize keyword statistics
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Keyword count distribution
ax1 = axes[0, 0]
ax1.hist(keyword_analysis['keyword_distribution'], bins=30, edgecolor='black', alpha=0.7)
ax1.axvline(keyword_analysis['avg_keywords_per_rule'], color='red', 
            linestyle='--', label=f"Mean: {keyword_analysis['avg_keywords_per_rule']:.1f}")
ax1.axvline(keyword_analysis['median_keywords_per_rule'], color='green', 
            linestyle='--', label=f"Median: {keyword_analysis['median_keywords_per_rule']:.0f}")
ax1.set_xlabel('Keywords per Rule')
ax1.set_ylabel('Number of Rules')
ax1.set_title('Distribution of Keywords per Rule')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Top keywords
ax2 = axes[0, 1]
top_20 = pd.Series(keyword_analysis['top_keywords'])
colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(top_20)))
bars = ax2.barh(range(len(top_20)), top_20.values, color=colors)
ax2.set_yticks(range(len(top_20)))
ax2.set_yticklabels(top_20.index, fontsize=9)
ax2.set_xlabel('Frequency')
ax2.set_title('Top 20 Most Frequent Keywords')
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)

# Keyword frequency distribution (log scale)
ax3 = axes[1, 0]
keyword_freqs = pd.Series(keyword_analysis['top_keywords']).values
ax3.loglog(range(1, len(keyword_freqs) + 1), keyword_freqs, 'bo-', markersize=4)
ax3.set_xlabel('Keyword Rank')
ax3.set_ylabel('Frequency')
ax3.set_title('Keyword Frequency Distribution (Log-Log Scale)')
ax3.grid(True, which="both", ls="-", alpha=0.2)

# Keyword diversity metrics
ax4 = axes[1, 1]
metrics = {
    'Unique\nKeywords': keyword_analysis['total_unique_keywords'],
    'Avg per\nRule': keyword_analysis['avg_keywords_per_rule'],
    'Coverage\n(%)': keyword_analysis['coverage'],
    'Singleton\n(%)': 100*keyword_analysis['singleton_keywords']/keyword_analysis['total_unique_keywords']
}
bars = ax4.bar(metrics.keys(), metrics.values())
ax4.set_title('Keyword Diversity Metrics')
ax4.set_ylabel('Value')
# Add value labels on bars
for bar, (key, val) in zip(bars, metrics.items()):
    height = bar.get_height()
    if 'Avg' in key:
        label = f'{val:.1f}'
    elif '%' in key:
        label = f'{val:.1f}%'
    else:
        label = f'{int(val):,}'
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            label, ha='center', va='bottom')
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('keyword_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 6. Embedding Analysis

# %% Embedding Quality Analysis
def analyze_embeddings(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze embedding quality and characteristics."""
    embeddings = []
    valid_count = 0
    invalid_count = 0
    
    for idx, emb in enumerate(df['embedding_parsed']):
        if emb is not None:
            if len(emb) == 1024:
                embeddings.append(emb)
                valid_count += 1
            else:
                invalid_count += 1
                print(f"Warning: Embedding at index {idx} has {len(emb)} dimensions (expected 1024)")
    
    if len(embeddings) == 0:
        return None
    
    embedding_matrix = np.stack(embeddings)
    
    # Calculate statistics
    norms = np.linalg.norm(embedding_matrix, axis=1)
    
    # Check if normalized
    is_normalized = np.allclose(norms, 1.0, rtol=1e-5)
    
    # Calculate pairwise similarities (sample for performance)
    sample_size = min(500, len(embeddings))
    np.random.seed(42)  # For reproducibility
    sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
    sample_matrix = embedding_matrix[sample_indices]
    
    # Normalize for cosine similarity
    sample_normalized = sample_matrix / np.linalg.norm(sample_matrix, axis=1, keepdims=True)
    similarities = np.dot(sample_normalized, sample_normalized.T)
    
    # Remove diagonal (self-similarity)
    mask = ~np.eye(similarities.shape[0], dtype=bool)
    similarities_flat = similarities[mask]
    
    # Component analysis
    embedding_means = embedding_matrix.mean(axis=0)
    embedding_stds = embedding_matrix.std(axis=0)
    
    return {
        'total_embeddings': valid_count,
        'invalid_embeddings': invalid_count,
        'coverage': 100 * valid_count / len(df),
        'embedding_dim': embedding_matrix.shape[1],
        'is_normalized': is_normalized,
        'norm_mean': norms.mean(),
        'norm_std': norms.std(),
        'norm_min': norms.min(),
        'norm_max': norms.max(),
        'similarity_mean': similarities_flat.mean(),
        'similarity_std': similarities_flat.std(),
        'similarity_min': similarities_flat.min(),
        'similarity_max': similarities_flat.max(),
        'similarity_percentiles': {
            '1%': np.percentile(similarities_flat, 1),
            '5%': np.percentile(similarities_flat, 5),
            '25%': np.percentile(similarities_flat, 25),
            '50%': np.percentile(similarities_flat, 50),
            '75%': np.percentile(similarities_flat, 75),
            '95%': np.percentile(similarities_flat, 95),
            '99%': np.percentile(similarities_flat, 99)
        },
        'component_mean_range': (embedding_means.min(), embedding_means.max()),
        'component_std_range': (embedding_stds.min(), embedding_stds.max()),
        'zero_components': (np.abs(embedding_means) < 1e-6).sum()
    }

embedding_analysis = analyze_embeddings(df)

if embedding_analysis:
    print("\n" + "=" * 80)
    print("EMBEDDING ANALYSIS (UAE-Large-V1, 1024-d)")
    print("=" * 80)
    print(f"Coverage: {embedding_analysis['coverage']:.1f}%")
    print(f"Valid embeddings: {embedding_analysis['total_embeddings']:,}")
    print(f"Invalid embeddings: {embedding_analysis['invalid_embeddings']}")
    print(f"Embedding dimension: {embedding_analysis['embedding_dim']}")
    print(f"L2-normalized: {embedding_analysis['is_normalized']}")
    print(f"\nNorm statistics:")
    print(f"  Mean: {embedding_analysis['norm_mean']:.6f}")
    print(f"  Std: {embedding_analysis['norm_std']:.6f}")
    print(f"  Range: [{embedding_analysis['norm_min']:.6f}, {embedding_analysis['norm_max']:.6f}]")
    
    print(f"\nCosine Similarity Distribution (n={500} sample):")
    print(f"  Mean: {embedding_analysis['similarity_mean']:.4f}")
    print(f"  Std: {embedding_analysis['similarity_std']:.4f}")
    print(f"  Range: [{embedding_analysis['similarity_min']:.4f}, {embedding_analysis['similarity_max']:.4f}]")
    print("  Percentiles:")
    for pct, val in embedding_analysis['similarity_percentiles'].items():
        print(f"    {pct:>3}: {val:.4f}")
    
    print(f"\nComponent Statistics:")
    print(f"  Mean range: [{embedding_analysis['component_mean_range'][0]:.6f}, "
          f"{embedding_analysis['component_mean_range'][1]:.6f}]")
    print(f"  Std range: [{embedding_analysis['component_std_range'][0]:.6f}, "
          f"{embedding_analysis['component_std_range'][1]:.6f}]")
    print(f"  Zero components: {embedding_analysis['zero_components']}/1024")

    # Visualize embedding characteristics
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Norm distribution
    ax1 = axes[0, 0]
    sample_embeddings = []
    for emb in df['embedding_parsed'][:1000]:
        if emb is not None and len(emb) == 1024:
            sample_embeddings.append(emb)
    
    if sample_embeddings:
        norms_plot = np.linalg.norm(np.stack(sample_embeddings), axis=1)
        ax1.hist(norms_plot, bins=50, edgecolor='black', alpha=0.7, color='blue')
        ax1.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Target (L2=1.0)')
        ax1.set_xlabel('L2 Norm')
        ax1.set_ylabel('Count')
        ax1.set_title('Embedding Norm Distribution')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
    
    # Similarity distribution
    ax2 = axes[0, 1]
    ax2.hist(similarities_flat, bins=50, edgecolor='black', alpha=0.7, color='green')
    ax2.axvline(embedding_analysis['similarity_mean'], color='red', 
                linestyle='--', label=f"Mean: {embedding_analysis['similarity_mean']:.3f}")
    ax2.axvline(0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Cosine Similarity')
    ax2.set_ylabel('Count')
    ax2.set_title('Pairwise Similarity Distribution')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Similarity heatmap (small sample)
    ax3 = axes[0, 2]
    sample_size_heat = min(50, len(embeddings))
    sample_indices_heat = np.random.choice(len(embeddings), sample_size_heat, replace=False)
    sample_matrix_heat = embedding_matrix[sample_indices_heat]
    sample_normalized_heat = sample_matrix_heat / np.linalg.norm(sample_matrix_heat, axis=1, keepdims=True)
    similarities_heat = np.dot(sample_normalized_heat, sample_normalized_heat.T)
    
    im = ax3.imshow(similarities_heat, cmap='coolwarm', vmin=-0.2, vmax=1.0)
    ax3.set_title(f'Similarity Matrix ({sample_size_heat}x{sample_size_heat} sample)')
    ax3.set_xlabel('Rule Index')
    ax3.set_ylabel('Rule Index')
    plt.colorbar(im, ax=ax3)
    
    # Component means distribution
    ax4 = axes[1, 0]
    embedding_means = embedding_matrix.mean(axis=0)
    ax4.hist(embedding_means, bins=50, edgecolor='black', alpha=0.7, color='purple')
    ax4.set_xlabel('Component Mean')
    ax4.set_ylabel('Count')
    ax4.set_title('Distribution of Component Means')
    ax4.axvline(0, color='black', linestyle='-', alpha=0.3)
    ax4.grid(axis='y', alpha=0.3)
    
    # Component standard deviations
    ax5 = axes[1, 1]
    embedding_stds = embedding_matrix.std(axis=0)
    ax5.hist(embedding_stds, bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax5.set_xlabel('Component Std Dev')
    ax5.set_ylabel('Count')
    ax5.set_title('Distribution of Component Std Deviations')
    ax5.grid(axis='y', alpha=0.3)
    
    # First few components visualization
    ax6 = axes[1, 2]
    n_components_show = 50
    ax6.plot(embedding_means[:n_components_show], 'b-', label='Mean', alpha=0.7)
    ax6.fill_between(range(n_components_show),
                     embedding_means[:n_components_show] - embedding_stds[:n_components_show],
                     embedding_means[:n_components_show] + embedding_stds[:n_components_show],
                     alpha=0.3, color='blue', label='±1 Std')
    ax6.set_xlabel('Component Index')
    ax6.set_ylabel('Value')
    ax6.set_title(f'First {n_components_show} Components (Mean ± Std)')
    ax6.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax6.legend()
    ax6.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('embedding_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## 7. Error Code Analysis

# %% Error Code Analysis
def analyze_error_codes(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze BANSTA and ISO error codes."""
    results = {}
    
    for code_field in ['bansta_error_code', 'iso_error_code']:
        if code_field in df.columns:
            codes = df[code_field].dropna()
            codes = codes[codes.astype(str).str.strip() != '']
            
            unique_codes = codes.value_counts()
            
            # Pattern analysis
            patterns = {
                'alphanumeric': codes.str.match(r'^[A-Z0-9]+$').sum(),
                'with_underscore': codes.str.contains('_').sum(),
                'with_dash': codes.str.contains('-').sum(),
                'numeric_only': codes.str.match(r'^\d+$').sum(),
                'starts_with_letter': codes.str.match(r'^[A-Z]').sum()
            }
            
            # Length analysis
            code_lengths = codes.str.len()
            
            results[code_field] = {
                'coverage': 100 * len(codes) / len(df),
                'unique_codes': len(unique_codes),
                'total_assignments': len(codes),
                'top_codes': unique_codes.head(15).to_dict(),
                'patterns': patterns,
                'length_stats': {
                    'mean': code_lengths.mean(),
                    'median': code_lengths.median(),
                    'min': code_lengths.min(),
                    'max': code_lengths.max()
                },
                'duplicate_rate': 100 * (len(codes) - len(unique_codes)) / len(codes)
            }
    
    return results

error_code_analysis = analyze_error_codes(df)

print("\n" + "=" * 80)
print("ERROR CODE ANALYSIS")
print("=" * 80)

for code_type, analysis in error_code_analysis.items():
    print(f"\n{code_type.upper().replace('_', ' ')}:")
    print(f"  Coverage: {analysis['coverage']:.1f}%")
    print(f"  Unique codes: {analysis['unique_codes']}")
    print(f"  Duplicate rate: {analysis['duplicate_rate']:.1f}%")
    print(f"  Length: mean={analysis['length_stats']['mean']:.1f}, "
          f"median={analysis['length_stats']['median']:.0f}, "
          f"range=[{analysis['length_stats']['min']:.0f}, {analysis['length_stats']['max']:.0f}]")
    print(f"  Top 5 codes:")
    for code, count in list(analysis['top_codes'].items())[:5]:
        pct = 100 * count / analysis['total_assignments']
        print(f"    - {code}: {count} ({pct:.1f}%)")

# Visualize error codes
if error_code_analysis:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, (code_type, analysis) in enumerate(error_code_analysis.items()):
        if idx < 2:
            ax = axes[idx]
            top_codes = pd.Series(analysis['top_codes'])
            colors = plt.cm.Set3(np.linspace(0, 1, len(top_codes)))
            ax.bar(range(len(top_codes)), top_codes.values, color=colors)
            ax.set_xticks(range(len(top_codes)))
            ax.set_xticklabels(top_codes.index, rotation=45, ha='right')
            ax.set_title(f'{code_type.upper().replace("_", " ")} Distribution\n'
                        f'Coverage: {analysis["coverage"]:.1f}%, '
                        f'Unique: {analysis["unique_codes"]}')
            ax.set_ylabel('Frequency')
            ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('error_code_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## 8. Temporal and Version Analysis

# %% Temporal Analysis
def analyze_temporal(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze temporal patterns and versioning in the corpus."""
    results = {}
    
    # Creation dates
    if 'created_at' in df.columns:
        created = df['created_at'].dropna()
        if not created.empty:
            results['creation'] = {
                'earliest': created.min(),
                'latest': created.max(),
                'span_days': (created.max() - created.min()).days,
                'rules_per_month': created.dt.to_period('M').value_counts().sort_index().to_dict(),
                'rules_per_year': created.dt.year.value_counts().sort_index().to_dict()
            }
    
    # Update dates
    if 'updated_at' in df.columns and 'created_at' in df.columns:
        updated = df['updated_at'].dropna()
        if not updated.empty:
            # Calculate update frequency
            both_dates = df[['created_at', 'updated_at']].dropna()
            update_deltas = (both_dates['updated_at'] - both_dates['created_at']).dt.days
            
            results['updates'] = {
                'earliest_update': updated.min(),
                'latest_update': updated.max(),
                'recently_updated_30d': (updated > updated.max() - pd.Timedelta(days=30)).sum(),
                'recently_updated_90d': (updated > updated.max() - pd.Timedelta(days=90)).sum(),
                'never_updated': (update_deltas == 0).sum(),
                'avg_days_to_update': update_deltas[update_deltas > 0].mean() if (update_deltas > 0).any() else 0,
                'median_days_to_update': update_deltas[update_deltas > 0].median() if (update_deltas > 0).any() else 0
            }
    
    # Version analysis
    if 'version_major' in df.columns and 'version_minor' in df.columns:
        version_data = df[['version_major', 'version_minor']].dropna()
        if not version_data.empty:
            # Create version string
            version_strings = version_data.apply(lambda x: f"{int(x['version_major'])}.{int(x['version_minor'])}", axis=1)
            version_counts = version_strings.value_counts()
            
            results['versioning'] = {
                'unique_versions': len(version_counts),
                'major_versions': df['version_major'].nunique(),
                'latest_major': df['version_major'].max(),
                'latest_minor': df.loc[df['version_major'] == df['version_major'].max(), 'version_minor'].max(),
                'version_distribution': version_counts.head(10).to_dict(),
                'most_common_version': version_counts.index[0] if len(version_counts) > 0 else None,
                'rules_at_latest': (version_strings == f"{df['version_major'].max()}.{df.loc[df['version_major'] == df['version_major'].max(), 'version_minor'].max()}").sum()
            }
    
    return results

temporal_analysis = analyze_temporal(df)

print("\n" + "=" * 80)
print("TEMPORAL AND VERSION ANALYSIS")
print("=" * 80)

if 'creation' in temporal_analysis:
    print(f"\nCreation Timeline:")
    print(f"  Earliest: {temporal_analysis['creation']['earliest']}")
    print(f"  Latest: {temporal_analysis['creation']['latest']}")
    print(f"  Span: {temporal_analysis['creation']['span_days']} days")
    print(f"  Rules per year:")
    for year, count in temporal_analysis['creation']['rules_per_year'].items():
        print(f"    {year}: {count}")

if 'updates' in temporal_analysis:
    print(f"\nUpdate Patterns:")
    print(f"  Recently updated (30 days): {temporal_analysis['updates']['recently_updated_30d']}")
    print(f"  Recently updated (90 days): {temporal_analysis['updates']['recently_updated_90d']}")
    print(f"  Never updated: {temporal_analysis['updates']['never_updated']}")
    print(f"  Avg days to first update: {temporal_analysis['updates']['avg_days_to_update']:.1f}")

if 'versioning' in temporal_analysis:
    print(f"\nVersioning:")
    print(f"  Unique versions: {temporal_analysis['versioning']['unique_versions']}")
    print(f"  Latest version: {temporal_analysis['versioning']['latest_major']}.{temporal_analysis['versioning']['latest_minor']}")
    print(f"  Most common: {temporal_analysis['versioning']['most_common_version']}")
    print(f"  Rules at latest version: {temporal_analysis['versioning']['rules_at_latest']}")

# %% [markdown]
# ## 9. Rule Code Analysis

# %% Rule Code Analysis
def analyze_rule_code(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze Kotlin rule code field."""
    if 'rule_code' not in df.columns:
        return None
    
    code_data = df['rule_code'].dropna()
    code_data = code_data[code_data.str.strip() != '']
    
    if code_data.empty:
        return None
    
    # Analyze code characteristics
    code_lengths = code_data.str.len()
    line_counts = code_data.str.count('\n') + 1
    
    # Kotlin patterns
    patterns = {
        'has_fun_keyword': code_data.str.contains(r'\bfun\b').sum(),
        'has_class_keyword': code_data.str.contains(r'\bclass\b').sum(),
        'has_if_statement': code_data.str.contains(r'\bif\b').sum(),
        'has_when_statement': code_data.str.contains(r'\bwhen\b').sum(),
        'has_return': code_data.str.contains(r'\breturn\b').sum(),
        'has_validate': code_data.str.contains(r'validat', case=False).sum(),
        'has_check': code_data.str.contains(r'check', case=False).sum(),
        'has_require': code_data.str.contains(r'require').sum()
    }
    
    return {
        'coverage': 100 * len(code_data) / len(df),
        'total_with_code': len(code_data),
        'avg_length': code_lengths.mean(),
        'median_length': code_lengths.median(),
        'max_length': code_lengths.max(),
        'avg_lines': line_counts.mean(),
        'median_lines': line_counts.median(),
        'max_lines': line_counts.max(),
        'patterns': patterns,
        'pattern_percentages': {k: 100*v/len(code_data) for k, v in patterns.items()}
    }

code_analysis = analyze_rule_code(df)

if code_analysis:
    print("\n" + "=" * 80)
    print("KOTLIN RULE CODE ANALYSIS")
    print("=" * 80)
    print(f"Coverage: {code_analysis['coverage']:.1f}%")
    print(f"Rules with code: {code_analysis['total_with_code']}")
    print(f"\nCode Metrics:")
    print(f"  Avg length: {code_analysis['avg_length']:.0f} chars")
    print(f"  Avg lines: {code_analysis['avg_lines']:.1f}")
    print(f"  Max lines: {code_analysis['max_lines']:.0f}")
    print(f"\nKotlin Pattern Presence:")
    for pattern, pct in code_analysis['pattern_percentages'].items():
        pattern_name = pattern.replace('_', ' ').replace('has ', '')
        print(f"  {pattern_name}: {pct:.1f}%")

# %% [markdown]
# ## 10. Data Quality Summary and Recommendations

# %% Data Quality Summary
def generate_quality_report(df: pd.DataFrame, 
                           completeness_df: pd.DataFrame,
                           keyword_analysis: Dict,
                           embedding_analysis: Dict) -> Dict[str, Any]:
    """Generate comprehensive data quality report."""
    
    # Critical fields for retrieval with thresholds
    critical_fields = {
        'rule_id': 100.0,
        'rule_name': 95.0,
        'keywords': 90.0,
        'embedding': 95.0,
        'llm_description': 85.0,
        'bansta_error_code': 80.0,
        'iso_error_code': 80.0
    }
    
    quality_scores = []
    issues = []
    
    for field, threshold in critical_fields.items():
        field_data = completeness_df[completeness_df['Field'] == field]
        if not field_data.empty:
            coverage = field_data['Non-Empty %'].iloc[0]
            quality_scores.append({
                'field': field,
                'coverage': coverage,
                'threshold': threshold,
                'status': 'PASS' if coverage >= threshold else 'FAIL',
                'gap': max(0, threshold - coverage)
            })
            
            if coverage < threshold:
                issues.append(f"{field}: {coverage:.1f}% < {threshold}% threshold (gap: {threshold-coverage:.1f}%)")
    
    # Calculate overall quality score (weighted by importance)
    weights = {'rule_id': 2.0, 'rule_name': 1.5, 'keywords': 1.5, 
               'embedding': 1.5, 'llm_description': 1.0, 
               'bansta_error_code': 0.8, 'iso_error_code': 0.8}
    
    weighted_scores = []
    total_weight = 0
    for score in quality_scores:
        weight = weights.get(score['field'], 1.0)
        weighted_scores.append(score['coverage'] * weight)
        total_weight += weight
    
    overall_score = sum(weighted_scores) / total_weight if total_weight > 0 else 0
    
    # Generate recommendations
    recommendations = []
    for score in quality_scores:
        if score['status'] == 'FAIL':
            field = score['field']
            gap = score['gap']
            missing_count = int(len(df) * gap / 100)
            
            if field == 'keywords':
                recommendations.append({
                    'priority': 'HIGH',
                    'action': f"Extract keywords for {missing_count} rules",
                    'impact': f"Reach {score['threshold']}% coverage target"
                })
            elif field == 'embedding':
                recommendations.append({
                    'priority': 'HIGH',
                    'action': f"Generate embeddings for {missing_count} rules using UAE-Large-V1",
                    'impact': f"Enable semantic search for all rules"
                })
            elif field == 'llm_description':
                recommendations.append({
                    'priority': 'MEDIUM',
                    'action': f"Generate LLM descriptions for {missing_count} rules",
                    'impact': f"Improve semantic search quality"
                })
    
    # Additional quality checks
    if keyword_analysis and keyword_analysis['avg_keywords_per_rule'] < 5:
        recommendations.append({
            'priority': 'MEDIUM',
            'action': "Enrich keyword extraction (current avg: {:.1f} keywords/rule)".format(
                keyword_analysis['avg_keywords_per_rule']),
            'impact': "Improve BM25 retrieval precision"
        })
    
    if embedding_analysis and not embedding_analysis['is_normalized']:
        recommendations.append({
            'priority': 'HIGH',
            'action': "L2-normalize all embeddings",
            'impact': "Ensure correct cosine similarity computation"
        })
    
    return {
        'overall_score': overall_score,
        'grade': 'A' if overall_score >= 95 else 'B' if overall_score >= 85 else 'C' if overall_score >= 75 else 'D',
        'field_scores': quality_scores,
        'issues': issues,
        'recommendations': recommendations,
        'summary': {
            'total_rules': len(df),
            'fields_analyzed': len(completeness_df),
            'critical_fields_passing': sum(1 for s in quality_scores if s['status'] == 'PASS'),
            'critical_fields_total': len(quality_scores)
        }
    }

quality_report = generate_quality_report(df, completeness_df, keyword_analysis, embedding_analysis)

print("\n" + "=" * 80)
print("DATA QUALITY REPORT")
print("=" * 80)
print(f"Overall Quality Score: {quality_report['overall_score']:.1f}% (Grade: {quality_report['grade']})")
print(f"Critical Fields: {quality_report['summary']['critical_fields_passing']}/{quality_report['summary']['critical_fields_total']} passing")

print("\nField Assessment:")
for score in quality_report['field_scores']:
    status_symbol = "✓" if score['status'] == 'PASS' else "✗"
    gap_str = f" (gap: {score['gap']:.1f}%)" if score['gap'] > 0 else ""
    print(f"  {status_symbol} {score['field']}: {score['coverage']:.1f}% "
          f"(target: {score['threshold']}%){gap_str}")

if quality_report['issues']:
    print("\nIssues Identified:")
    for issue in quality_report['issues']:
        print(f"  - {issue}")

if quality_report['recommendations']:
    print("\nPrioritized Recommendations:")
    for rec in sorted(quality_report['recommendations'], 
                     key=lambda x: {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}[x['priority']]):
        print(f"  [{rec['priority']}] {rec['action']}")
        print(f"         Impact: {rec['impact']}")

# Final visualization: Quality Dashboard
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Overall quality gauge
ax1 = axes[0, 0]
score = quality_report['overall_score']
colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
color_idx = min(4, int(score / 20))
ax1.pie([score, 100-score], colors=[colors[color_idx], 'lightgray'],
        startangle=90, counterclock=False)
ax1.text(0, 0, f"{score:.1f}%\nGrade: {quality_report['grade']}", 
         ha='center', va='center', fontsize=20, fontweight='bold')
ax1.set_title('Overall Quality Score', fontsize=14, fontweight='bold')

# Field coverage bars
ax2 = axes[0, 1]
field_data = pd.DataFrame(quality_report['field_scores'])
colors_bars = ['green' if s == 'PASS' else 'red' for s in field_data['status']]
bars = ax2.barh(field_data['field'], field_data['coverage'], color=colors_bars, alpha=0.7)
ax2.barh(field_data['field'], field_data['threshold'], color='black', alpha=0.3, height=0.3)
ax2.set_xlabel('Coverage (%)')
ax2.set_title('Critical Field Coverage vs Thresholds', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 105)
for i, (cov, thresh) in enumerate(zip(field_data['coverage'], field_data['threshold'])):
    ax2.text(cov + 1, i, f'{cov:.1f}%', va='center')
ax2.grid(axis='x', alpha=0.3)

# Key metrics
ax3 = axes[1, 0]
metrics = {
    'Total Rules': len(df),
    'Unique Keywords': keyword_analysis['total_unique_keywords'] if keyword_analysis else 0,
    'Avg Keywords/Rule': keyword_analysis['avg_keywords_per_rule'] if keyword_analysis else 0,
    'Embeddings Valid': embedding_analysis['total_embeddings'] if embedding_analysis else 0,
    'L2 Normalized': 'Yes' if embedding_analysis and embedding_analysis['is_normalized'] else 'No'
}
y_pos = np.arange(len(metrics))
ax3.axis('off')
table_data = [[k, f"{v:,}" if isinstance(v, int) else f"{v:.1f}" if isinstance(v, float) else v] 
              for k, v in metrics.items()]
table = ax3.table(cellText=table_data, colLabels=['Metric', 'Value'],
                  cellLoc='left', loc='center', colWidths=[0.6, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)
ax3.set_title('Key Corpus Metrics', fontsize=14, fontweight='bold', pad=20)

# Recommendations
ax4 = axes[1, 1]
ax4.axis('off')
rec_text = "Priority Actions:\n\n"
for i, rec in enumerate(sorted(quality_report['recommendations'][:4], 
                              key=lambda x: {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}[x['priority']]), 1):
    rec_text += f"{i}. [{rec['priority']}] {rec['action']}\n"
ax4.text(0.1, 0.9, rec_text, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace')
ax4.set_title('Top Recommendations', fontsize=14, fontweight='bold')

plt.suptitle('Corpus Quality Dashboard', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('quality_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 11. Export Summary

# %% Export Summary
import json
from datetime import datetime

summary = {
    'metadata': {
        'analysis_date': datetime.now().isoformat(),
        'corpus_size': len(df),
        'field_count': len(df.columns)
    },
    'completeness': {
        'summary': completeness_df[['Field', 'Non-Empty %']].to_dict('records'),
        'critical_coverage': {field: completeness_df[completeness_df['Field'] == field]['Non-Empty %'].iloc[0] 
                             for field in ['rule_id', 'rule_name', 'keywords', 'embedding', 'llm_description']
                             if field in completeness_df['Field'].values}
    },
    'tags': {field: {
        'coverage': analysis['coverage'],
        'unique_values': analysis['unique_values'],
        'entropy': analysis['entropy']
    } for field, analysis in tag_analysis.items()},
    'keywords': {
        'coverage': keyword_analysis['coverage'],
        'unique_keywords': keyword_analysis['total_unique_keywords'],
        'avg_per_rule': keyword_analysis['avg_keywords_per_rule'],
        'median_per_rule': keyword_analysis['median_keywords_per_rule']
    } if keyword_analysis else None,
    'embeddings': {
        'coverage': embedding_analysis['coverage'],
        'dimension': embedding_analysis['embedding_dim'],
        'is_normalized': embedding_analysis['is_normalized'],
        'similarity_mean': embedding_analysis['similarity_mean'],
        'similarity_std': embedding_analysis['similarity_std']
    } if embedding_analysis else None,
    'quality': {
        'overall_score': quality_report['overall_score'],
        'grade': quality_report['grade'],
        'issues_count': len(quality_report['issues']),
        'recommendations_count': len(quality_report['recommendations'])
    }
}

# Save summary
with open('corpus_analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("Summary saved to: corpus_analysis_summary.json")
print("\nGenerated visualizations:")
print("  - field_completeness.png")
print("  - tag_distributions.png")
print("  - text_field_analysis.png")
print("  - keyword_analysis.png")
print("  - embedding_analysis.png")
print("  - error_code_analysis.png")
print("  - quality_dashboard.png")
print("\nTotal corpus size: {:,} rules".format(len(df)))
print("Overall quality grade: {}".format(quality_report['grade']))