import torch
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import csv
import geoopt
from transformers import AutoTokenizer
from utils.logger import get_logger

logger = get_logger(__name__)

# Load Hyperbolic Model
class HyperbolicDualEncoder(torch.nn.Module):
    def __init__(self, vocab_size, dim=128, c=1.0):
        super().__init__()
        self.manifold = geoopt.PoincareBall(c=c)
        self.emb = geoopt.ManifoldParameter(
            self.manifold.random_normal((vocab_size, dim)), 
            manifold=self.manifold
        )
    
    def forward(self, input_ids):
        token_vecs = self.emb[input_ids]
        tang = self.manifold.logmap0(token_vecs)
        mean_tang = tang.mean(dim=1)
        sent = self.manifold.expmap0(mean_tang)
        return self.manifold.projx(sent)

def load_data(queries_path="wordNet/queries.tsv", corpus_path="wordNet/corpus.tsv"):
    """Load queries and corpus."""
    corpus = {}
    with open(corpus_path, "r") as fc:
        for row in csv.reader(fc, delimiter='\t'):
            corpus[row[0]] = row[1]
    
    queries = []
    relevance = []
    with open(queries_path, "r") as fq:
        for row in csv.reader(fq, delimiter='\t'):
            qid, qtext, pos_id = row
            queries.append(qtext)
            relevance.append(int(pos_id))
    
    corpus_texts = [corpus[str(i)] for i in range(len(corpus))]
    return queries, corpus_texts, relevance

def encode_euclidean(model, texts):
    """Encode texts using Euclidean model."""
    return model.encode(texts, convert_to_tensor=True)

def encode_hyperbolic(model, tokenizer, texts, device):
    """Encode texts using Hyperbolic model."""
    embeddings = []
    model.eval()
    with torch.no_grad():
        for text in texts:
            tok = tokenizer(text, return_tensors='pt', padding='max_length', 
                          truncation=True, max_length=32)
            input_ids = tok["input_ids"].to(device)
            emb = model(input_ids)
            embeddings.append(emb)
    return torch.cat(embeddings, dim=0)

def compute_retrieval_metrics(query_embs, corpus_embs, relevance, model_type, manifold=None):
    """Compute comprehensive retrieval metrics."""
    n_queries = len(query_embs)
    mrr_scores = []
    recall_at_1, recall_at_5, recall_at_10, recall_at_20 = [], [], [], []
    precision_at_1, precision_at_5, precision_at_10 = [], [], []
    ndcg_at_5, ndcg_at_10, ndcg_at_20 = [], [], []
    map_scores = []
    hit_rate_at_10 = []
    
    for i in tqdm(range(n_queries), desc=f"Evaluating {model_type}"):
        if model_type == "Hyperbolic":
            q_emb = query_embs[i].unsqueeze(0)
            distances = manifold.dist(q_emb, corpus_embs).squeeze()
            scores = -distances
            ranked = torch.argsort(scores, descending=True).cpu().numpy()
        else:
            q_emb = query_embs[i].unsqueeze(0)
            scores = util.cos_sim(q_emb, corpus_embs).squeeze()
            ranked = torch.argsort(scores, descending=True).cpu().numpy()
        
        relevant_idx = relevance[i]
        rank = np.where(ranked == relevant_idx)[0][0] + 1
        
        # MRR
        mrr_scores.append(1.0 / rank)
        
        # Recall@K
        recall_at_1.append(1 if rank <= 1 else 0)
        recall_at_5.append(1 if rank <= 5 else 0)
        recall_at_10.append(1 if rank <= 10 else 0)
        recall_at_20.append(1 if rank <= 20 else 0)
        
        # Precision@K
        precision_at_1.append(1 if rank <= 1 else 0)
        precision_at_5.append(1/5 if rank <= 5 else 0)
        precision_at_10.append(1/10 if rank <= 10 else 0)
        
        # NDCG@K
        def dcg_at_k(rank, k):
            if rank > k:
                return 0
            return 1.0 / np.log2(rank + 1)
        
        ndcg_at_5.append(dcg_at_k(rank, 5) / dcg_at_k(1, 5))
        ndcg_at_10.append(dcg_at_k(rank, 10) / dcg_at_k(1, 10))
        ndcg_at_20.append(dcg_at_k(rank, 20) / dcg_at_k(1, 20))
        
        # MAP (Mean Average Precision)
        ap = 1.0 / rank if rank <= len(corpus_embs) else 0
        map_scores.append(ap)
        
        # Hit Rate@10
        hit_rate_at_10.append(1 if rank <= 10 else 0)
    
    return {
        'MRR': np.mean(mrr_scores),
        'MAP': np.mean(map_scores),
        'Recall@1': np.mean(recall_at_1),
        'Recall@5': np.mean(recall_at_5),
        'Recall@10': np.mean(recall_at_10),
        'Recall@20': np.mean(recall_at_20),
        'Precision@1': np.mean(precision_at_1),
        'Precision@5': np.mean(precision_at_5),
        'Precision@10': np.mean(precision_at_10),
        'NDCG@5': np.mean(ndcg_at_5),
        'NDCG@10': np.mean(ndcg_at_10),
        'NDCG@20': np.mean(ndcg_at_20),
        'Hit@10': np.mean(hit_rate_at_10)
    }

def retrieve_samples(query_embs, corpus_embs, queries, corpus, relevance, 
                    model_type, manifold=None, n_samples=5):
    """Retrieve and display sample results."""
    results = []
    
    for i in range(min(n_samples, len(queries))):
        if model_type == "Hyperbolic":
            q_emb = query_embs[i].unsqueeze(0)
            distances = manifold.dist(q_emb, corpus_embs).squeeze()
            scores = -distances
            top_k = torch.argsort(scores, descending=True)[:5].cpu().numpy()
        else:
            q_emb = query_embs[i].unsqueeze(0)
            scores = util.cos_sim(q_emb, corpus_embs).squeeze()
            top_k = torch.argsort(scores, descending=True)[:5].cpu().numpy()
        
        results.append({
            'query': queries[i],
            'relevant_doc': corpus[relevance[i]],
            'top_5_retrieved': [corpus[idx] for idx in top_k],
            'relevant_in_top_5': relevance[i] in top_k
        })
    
    return results

def plot_metrics(euc_metrics, hyp_metrics):
    """Plot comprehensive comparison metrics."""
    # Group metrics by category
    recall_metrics = ['Recall@1', 'Recall@5', 'Recall@10', 'Recall@20']
    precision_metrics = ['Precision@1', 'Precision@5', 'Precision@10']
    ndcg_metrics = ['NDCG@5', 'NDCG@10', 'NDCG@20']
    overall_metrics = ['MRR', 'MAP', 'Hit@10']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Recall metrics
    x = np.arange(len(recall_metrics))
    width = 0.35
    axes[0, 0].bar(x - width/2, [euc_metrics[m] for m in recall_metrics], 
                   width, label='Euclidean', alpha=0.8, color='steelblue')
    axes[0, 0].bar(x + width/2, [hyp_metrics[m] for m in recall_metrics], 
                   width, label='Hyperbolic', alpha=0.8, color='coral')
    axes[0, 0].set_title('Recall Metrics', fontsize=13, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(recall_metrics, rotation=15)
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Precision metrics
    x = np.arange(len(precision_metrics))
    axes[0, 1].bar(x - width/2, [euc_metrics[m] for m in precision_metrics], 
                   width, label='Euclidean', alpha=0.8, color='steelblue')
    axes[0, 1].bar(x + width/2, [hyp_metrics[m] for m in precision_metrics], 
                   width, label='Hyperbolic', alpha=0.8, color='coral')
    axes[0, 1].set_title('Precision Metrics', fontsize=13, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(precision_metrics, rotation=15)
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: NDCG metrics
    x = np.arange(len(ndcg_metrics))
    axes[1, 0].bar(x - width/2, [euc_metrics[m] for m in ndcg_metrics], 
                   width, label='Euclidean', alpha=0.8, color='steelblue')
    axes[1, 0].bar(x + width/2, [hyp_metrics[m] for m in ndcg_metrics], 
                   width, label='Hyperbolic', alpha=0.8, color='coral')
    axes[1, 0].set_title('NDCG Metrics', fontsize=13, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(ndcg_metrics, rotation=15)
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Overall metrics
    x = np.arange(len(overall_metrics))
    axes[1, 1].bar(x - width/2, [euc_metrics[m] for m in overall_metrics], 
                   width, label='Euclidean', alpha=0.8, color='steelblue')
    axes[1, 1].bar(x + width/2, [hyp_metrics[m] for m in overall_metrics], 
                   width, label='Hyperbolic', alpha=0.8, color='coral')
    axes[1, 1].set_title('Overall Metrics', fontsize=13, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(overall_metrics, rotation=15)
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Comprehensive Model Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: metrics_comparison.png")

def plot_distance_distribution(euc_dists, hyp_dists):
    """Plot distance distributions and rank distributions."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Distance distributions
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(euc_dists, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_title('Euclidean Distance Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Distance')
    ax1.set_ylabel('Frequency')
    ax1.grid(alpha=0.3)
    ax1.axvline(np.mean(euc_dists), color='red', linestyle='--', 
                label=f'Mean: {np.mean(euc_dists):.3f}')
    ax1.legend()
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(hyp_dists, bins=50, alpha=0.7, color='coral', edgecolor='black')
    ax2.set_title('Hyperbolic Distance Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Distance')
    ax2.set_ylabel('Frequency')
    ax2.grid(alpha=0.3)
    ax2.axvline(np.mean(hyp_dists), color='red', linestyle='--', 
                label=f'Mean: {np.mean(hyp_dists):.3f}')
    ax2.legend()
    
    # Box plots
    ax3 = fig.add_subplot(gs[1, :])
    box_data = [euc_dists, hyp_dists]
    bp = ax3.boxplot(box_data, labels=['Euclidean', 'Hyperbolic'], 
                     patch_artist=True, showmeans=True)
    colors = ['steelblue', 'coral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_title('Distance Distribution Comparison (Box Plot)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Distance')
    ax3.grid(axis='y', alpha=0.3)
    
    # Statistics text
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    stats_text = f"""
    Distance Statistics:
    
    Euclidean:                                 Hyperbolic:
    Mean:      {np.mean(euc_dists):.4f}        Mean:      {np.mean(hyp_dists):.4f}
    Median:    {np.median(euc_dists):.4f}        Median:    {np.median(hyp_dists):.4f}
    Std Dev:   {np.std(euc_dists):.4f}        Std Dev:   {np.std(hyp_dists):.4f}
    Min:       {np.min(euc_dists):.4f}        Min:       {np.min(hyp_dists):.4f}
    Max:       {np.max(euc_dists):.4f}        Max:       {np.max(hyp_dists):.4f}
    """
    ax4.text(0.5, 0.5, stats_text, fontsize=11, family='monospace',
             ha='center', va='center', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Distance Analysis', fontsize=14, fontweight='bold')
    plt.savefig('distance_distributions.png', dpi=300, bbox_inches='tight')
    print("Saved: distance_distributions.png")

def plot_rank_distribution(euc_q_embs, euc_c_embs, hyp_q_embs, hyp_c_embs, relevance, manifold):
    """Plot rank distribution for relevant documents."""
    euc_ranks = []
    hyp_ranks = []
    
    for i in range(len(relevance)):
        # Euclidean
        q_emb = euc_q_embs[i].unsqueeze(0)
        scores = util.cos_sim(q_emb, euc_c_embs).squeeze()
        ranked = torch.argsort(scores, descending=True).cpu().numpy()
        rank = np.where(ranked == relevance[i])[0][0] + 1
        euc_ranks.append(rank)
        
        # Hyperbolic
        q_emb = hyp_q_embs[i].unsqueeze(0)
        distances = manifold.dist(q_emb, hyp_c_embs).squeeze()
        scores = -distances
        ranked = torch.argsort(scores, descending=True).cpu().numpy()
        rank = np.where(ranked == relevance[i])[0][0] + 1
        hyp_ranks.append(rank)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Rank histograms
    max_rank = max(max(euc_ranks), max(hyp_ranks))
    bins = min(50, max_rank)
    
    axes[0].hist(euc_ranks, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].set_title('Euclidean: Rank Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Rank of Relevant Document')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(np.median(euc_ranks), color='red', linestyle='--', 
                   label=f'Median: {np.median(euc_ranks):.1f}')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].hist(hyp_ranks, bins=bins, alpha=0.7, color='coral', edgecolor='black')
    axes[1].set_title('Hyperbolic: Rank Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Rank of Relevant Document')
    axes[1].set_ylabel('Frequency')
    axes[1].axvline(np.median(hyp_ranks), color='red', linestyle='--', 
                   label=f'Median: {np.median(hyp_ranks):.1f}')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Cumulative rank comparison
    euc_sorted = np.sort(euc_ranks)
    hyp_sorted = np.sort(hyp_ranks)
    euc_cumulative = np.arange(1, len(euc_sorted) + 1) / len(euc_sorted)
    hyp_cumulative = np.arange(1, len(hyp_sorted) + 1) / len(hyp_sorted)
    
    axes[2].plot(euc_sorted, euc_cumulative, label='Euclidean', 
                linewidth=2, color='steelblue')
    axes[2].plot(hyp_sorted, hyp_cumulative, label='Hyperbolic', 
                linewidth=2, color='coral')
    axes[2].set_title('Cumulative Rank Distribution', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Rank')
    axes[2].set_ylabel('Cumulative Proportion')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    axes[2].set_xlim(0, min(100, max_rank))
    
    plt.tight_layout()
    plt.savefig('rank_distributions.png', dpi=300, bbox_inches='tight')
    print("Saved: rank_distributions.png")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading data...")
    queries, corpus, relevance = load_data()
    print(f"Loaded {len(queries)} queries and {len(corpus)} documents\n")
    
    # Load Euclidean model
    print("Loading Euclidean model...")
    euc_model = SentenceTransformer("models/wordNet/euclidean/")
    
    # Load Hyperbolic model
    print("Loading Hyperbolic model...")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    hyp_model = HyperbolicDualEncoder(vocab_size=tokenizer.vocab_size, dim=128).to(device)
    hyp_model.load_state_dict(torch.load("models/wordNet/hyperbolic/model.pth", map_location=device))
    
    # Encode
    print("\nEncoding with Euclidean model...")
    euc_query_embs = encode_euclidean(euc_model, queries)
    euc_corpus_embs = encode_euclidean(euc_model, corpus)
    
    print("Encoding with Hyperbolic model...")
    hyp_query_embs = encode_hyperbolic(hyp_model, tokenizer, queries, device)
    hyp_corpus_embs = encode_hyperbolic(hyp_model, tokenizer, corpus, device)
    
    # Compute metrics
    print("\n" + "="*80)
    print("COMPREHENSIVE RETRIEVAL METRICS")
    print("="*80)
    
    euc_metrics = compute_retrieval_metrics(euc_query_embs, euc_corpus_embs, 
                                           relevance, "Euclidean")
    print("\nEuclidean Model:")
    print("-" * 40)
    print("Ranking Metrics:")
    print(f"  {'MRR':20s}: {euc_metrics['MRR']:.4f}")
    print(f"  {'MAP':20s}: {euc_metrics['MAP']:.4f}")
    print(f"  {'Hit@10':20s}: {euc_metrics['Hit@10']:.4f}")
    print("\nRecall Metrics:")
    print(f"  {'Recall@1':20s}: {euc_metrics['Recall@1']:.4f}")
    print(f"  {'Recall@5':20s}: {euc_metrics['Recall@5']:.4f}")
    print(f"  {'Recall@10':20s}: {euc_metrics['Recall@10']:.4f}")
    print(f"  {'Recall@20':20s}: {euc_metrics['Recall@20']:.4f}")
    print("\nPrecision Metrics:")
    print(f"  {'Precision@1':20s}: {euc_metrics['Precision@1']:.4f}")
    print(f"  {'Precision@5':20s}: {euc_metrics['Precision@5']:.4f}")
    print(f"  {'Precision@10':20s}: {euc_metrics['Precision@10']:.4f}")
    print("\nNDCG Metrics:")
    print(f"  {'NDCG@5':20s}: {euc_metrics['NDCG@5']:.4f}")
    print(f"  {'NDCG@10':20s}: {euc_metrics['NDCG@10']:.4f}")
    print(f"  {'NDCG@20':20s}: {euc_metrics['NDCG@20']:.4f}")
    
    hyp_metrics = compute_retrieval_metrics(hyp_query_embs, hyp_corpus_embs, 
                                           relevance, "Hyperbolic", hyp_model.manifold)
    print("\n" + "="*80)
    print("\nHyperbolic Model:")
    print("-" * 40)
    print("Ranking Metrics:")
    print(f"  {'MRR':20s}: {hyp_metrics['MRR']:.4f}")
    print(f"  {'MAP':20s}: {hyp_metrics['MAP']:.4f}")
    print(f"  {'Hit@10':20s}: {hyp_metrics['Hit@10']:.4f}")
    print("\nRecall Metrics:")
    print(f"  {'Recall@1':20s}: {hyp_metrics['Recall@1']:.4f}")
    print(f"  {'Recall@5':20s}: {hyp_metrics['Recall@5']:.4f}")
    print(f"  {'Recall@10':20s}: {hyp_metrics['Recall@10']:.4f}")
    print(f"  {'Recall@20':20s}: {hyp_metrics['Recall@20']:.4f}")
    print("\nPrecision Metrics:")
    print(f"  {'Precision@1':20s}: {hyp_metrics['Precision@1']:.4f}")
    print(f"  {'Precision@5':20s}: {hyp_metrics['Precision@5']:.4f}")
    print(f"  {'Precision@10':20s}: {hyp_metrics['Precision@10']:.4f}")
    print("\nNDCG Metrics:")
    print(f"  {'NDCG@5':20s}: {hyp_metrics['NDCG@5']:.4f}")
    print(f"  {'NDCG@10':20s}: {hyp_metrics['NDCG@10']:.4f}")
    print(f"  {'NDCG@20':20s}: {hyp_metrics['NDCG@20']:.4f}")
    
    # Plot metrics
    plot_metrics(euc_metrics, hyp_metrics)
    
    # Additional visualization: Rank distribution
    plot_rank_distribution(euc_query_embs, euc_corpus_embs, hyp_query_embs, 
                          hyp_corpus_embs, relevance, hyp_model.manifold)
    
    # Distance distributions
    print("\nComputing distance distributions...")
    euc_dists = []
    hyp_dists = []
    for i in range(min(100, len(queries))):
        euc_d = torch.norm(euc_query_embs[i].unsqueeze(0) - euc_corpus_embs, dim=1)
        euc_dists.extend(euc_d.cpu().numpy())
        
        hyp_d = hyp_model.manifold.dist(hyp_query_embs[i].unsqueeze(0), hyp_corpus_embs)
        hyp_dists.extend(hyp_d.cpu().numpy())
    
    plot_distance_distribution(euc_dists, hyp_dists)
    
    # Sample retrievals
    print("\n" + "="*60)
    print("SAMPLE RETRIEVALS")
    print("="*60)
    
    euc_samples = retrieve_samples(euc_query_embs, euc_corpus_embs, queries, 
                                   corpus, relevance, "Euclidean", n_samples=3)
    hyp_samples = retrieve_samples(hyp_query_embs, hyp_corpus_embs, queries, 
                                   corpus, relevance, "Hyperbolic", 
                                   hyp_model.manifold, n_samples=3)
    
    for i in range(3):
        print(f"\n{'='*60}")
        print(f"QUERY {i+1}: {queries[i][:100]}...")
        print(f"{'='*60}")
        
        print(f"\nRELEVANT DOCUMENT:")
        print(f"  {corpus[relevance[i]][:200]}...")
        
        print(f"\n[EUCLIDEAN] Top-3 Retrieved:")
        for j, doc in enumerate(euc_samples[i]['top_5_retrieved'][:3]):
            marker = "✓" if doc == corpus[relevance[i]] else "✗"
            print(f"  {j+1}. {marker} {doc[:150]}...")
        
        print(f"\n[HYPERBOLIC] Top-3 Retrieved:")
        for j, doc in enumerate(hyp_samples[i]['top_5_retrieved'][:3]):
            marker = "✓" if doc == corpus[relevance[i]] else "✗"
            print(f"  {j+1}. {marker} {doc[:150]}...")
    
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*80)
    
    # Create comparison table
    all_metrics = ['MRR', 'MAP', 'Hit@10', 'Recall@1', 'Recall@5', 'Recall@10', 
                   'Recall@20', 'Precision@1', 'Precision@5', 'Precision@10',
                   'NDCG@5', 'NDCG@10', 'NDCG@20']
    
    print(f"\n{'Metric':<20} {'Euclidean':>12} {'Hyperbolic':>12} {'Improvement':>12}")
    print("-" * 60)
    for metric in all_metrics:
        euc_val = euc_metrics[metric]
        hyp_val = hyp_metrics[metric]
        improvement = ((hyp_val - euc_val) / euc_val * 100) if euc_val > 0 else 0
        print(f"{metric:<20} {euc_val:>12.4f} {hyp_val:>12.4f} {improvement:>11.2f}%")
    
    # Statistical significance indicators
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    wins_hyp = sum(1 for m in all_metrics if hyp_metrics[m] > euc_metrics[m])
    wins_euc = sum(1 for m in all_metrics if euc_metrics[m] > hyp_metrics[m])
    
    print(f"\nHyperbolic wins: {wins_hyp}/{len(all_metrics)} metrics")
    print(f"Euclidean wins: {wins_euc}/{len(all_metrics)} metrics")
    
    avg_improvement = np.mean([((hyp_metrics[m] - euc_metrics[m]) / euc_metrics[m] * 100) 
                                if euc_metrics[m] > 0 else 0 for m in all_metrics])
    print(f"\nAverage improvement (Hyperbolic over Euclidean): {avg_improvement:+.2f}%")

if __name__ == "__main__":
    main()