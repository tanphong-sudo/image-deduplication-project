import json
from pathlib import Path

def print_results():
    # Load evaluation
    eval_file = Path("data/processed/evaluation_full.json")
    if not eval_file.exists():
        print("❌ Evaluation file not found!")
        return
    
    with open(eval_file) as f:
        data = json.load(f)
    
    # Load clusters
    clusters_file = Path("data/processed/faiss_clusters.json")
    clusters_data = {}
    if clusters_file.exists():
        with open(clusters_file) as f:
            clusters_data = json.load(f)
    
    # Print header
    print("\n" + "="*70)
    print("📊 IMAGE DEDUPLICATION - EVALUATION RESULTS")
    print("="*70)
    
    # Metrics
    print("\n🎯 CLUSTERING METRICS")
    print("-"*70)
    print(f"  Precision:        {data['precision']:.4f} ({data['precision']*100:.2f}%)")
    print(f"  Recall:           {data['recall']:.4f} ({data['recall']*100:.2f}%)")
    print(f"  True Positives:   {data['tp']:,}")
    print(f"  False Positives:  {data['fp']:,}")
    print(f"  False Negatives:  {data['fn']:,}")
    
    # Clusters info
    if clusters_data:
        print(f"\n📦 CLUSTERS")
        print("-"*70)
        print(f"  Total clusters: {len(clusters_data.get('clusters', []))}")
        for i, cluster in enumerate(clusters_data.get('clusters', [])):
            obj_name = cluster[0].split('/')[-1].split('__')[0]
            print(f"    Cluster {i}: {obj_name:5s} - {len(cluster):3d} images")
        
        print(f"\n⭐ REPRESENTATIVES")
        print("-"*70)
        reps = clusters_data.get('representatives', {})
        for cid, rep in reps.items():
            rep_name = rep.split("/")[-1]
            print(f"    Cluster {cid}: {rep_name}")
    
    # Performance
    print(f"\n⏱️  PERFORMANCE")
    print("-"*70)
    timings = data.get('timings', {})
    total_time = sum(timings.values())
    for k, v in timings.items():
        pct = (v / total_time * 100) if total_time > 0 else 0
        print(f"  {k:25s}: {v:8.4f}s ({pct:5.1f}%)")
    print(f"  {'TOTAL':25s}: {total_time:8.4f}s")
    
    # Memory
    print(f"\n💾 MEMORY USAGE")
    print("-"*70)
    memory = data.get('memory', {})
    total_mem = sum(memory.values())
    for k, v in memory.items():
        print(f"  {k:25s}: {v:8.2f} MB")
    print(f"  {'PEAK TOTAL':25s}: {total_mem:8.2f} MB")
    
    print("\n" + "="*70)
    print("✅ Evaluation complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    print_results()
