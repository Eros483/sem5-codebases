import requests
import csv
import os
from collections import defaultdict

def download_wordnet_hierarchy():
    """
    Download and transform WordNet hierarchy into MSMarco-style format.
    Uses NLTK's WordNet data.
    """
    try:
        import nltk
        from nltk.corpus import wordnet as wn
    except ImportError:
        print("Installing required packages...")
        os.system("pip install nltk")
        import nltk
        from nltk.corpus import wordnet as wn
    
    # Download WordNet data
    print("Downloading WordNet data...")
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    
    print("Processing WordNet hierarchy...")
    
    # Create output directory
    os.makedirs("wordNet", exist_ok=True)
    
    # Dictionary to store all synsets and their hypernyms
    corpus = {}  # synset_id -> definition/name
    queries = []  # (child_synset, parent_synset_id)
    
    synset_to_id = {}
    current_id = 0
    
    # Process all noun synsets (they have the clearest hierarchies)
    all_synsets = list(wn.all_synsets('n'))  # 'n' for nouns
    
    print(f"Found {len(all_synsets)} noun synsets...")
    
    for synset in all_synsets:
        # Assign ID to this synset if not already done
        if synset.name() not in synset_to_id:
            synset_to_id[synset.name()] = current_id
            corpus[current_id] = synset.definition()  # Use definition as the "passage"
            current_id += 1
        
        child_id = synset_to_id[synset.name()]
        
        # Get hypernyms (parent concepts)
        hypernyms = synset.hypernyms()
        
        for parent_synset in hypernyms:
            # Assign ID to parent if not already done
            if parent_synset.name() not in synset_to_id:
                synset_to_id[parent_synset.name()] = current_id
                corpus[current_id] = parent_synset.definition()
                current_id += 1
            
            parent_id = synset_to_id[parent_synset.name()]
            
            # Create training example: child concept -> parent concept
            # Use the child's lemma names as "query"
            child_name = ", ".join(synset.lemma_names()[:3])  # First 3 names
            queries.append((child_name, parent_id))
    
    print(f"Generated {len(queries)} hierarchical relationships...")
    
    # Write corpus.tsv
    print("Writing corpus.tsv...")
    with open("wordNet/corpus.tsv", "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        for corpus_id, definition in corpus.items():
            writer.writerow([corpus_id, definition])
    
    # Write queries.tsv
    print("Writing queries.tsv...")
    with open("wordNet/queries.tsv", "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        for idx, (child_name, parent_id) in enumerate(queries):
            writer.writerow([idx, child_name, parent_id])
    
    print(f"\n✅ Done!")
    print(f"   - Corpus: {len(corpus)} concepts")
    print(f"   - Queries: {len(queries)} hierarchical pairs")
    print(f"   - Files saved to wordNet/corpus.tsv and wordNet/queries.tsv")
    
    # Print sample
    print("\nSample data:")
    print("Corpus (first 3):")
    for i in range(min(3, len(corpus))):
        print(f"  {i}: {corpus[i][:80]}...")
    print("\nQueries (first 3):")
    for i in range(min(3, len(queries))):
        print(f"  {i}: {queries[i][0]} -> parent_id:{queries[i][1]}")

if __name__ == "__main__":
    download_wordnet_hierarchy()