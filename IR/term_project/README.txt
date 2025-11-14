# <div align="center">Experiment with Hyperbolic and Euclidean space Embeddings</div>
This project investigates performance for encoder models built in the Euclidean space versus built in Hyperbolic space.

We build two parallel pipelines and then a testing pipeline to achieve the same.
Both tests were conducted on training MiniLM-L6-v2

## Expectations
For instance if our vocabulary consisted of Dog, Mammal, Animal, Chihuahua, and Bird
- Euclidean would say dog is similarly close to mammal and animal
- Hyperbolic would say dog is much closer to mammal, than to animal

We expect Euclidean models to perform very well on flat datasets, and for Hyperbolic models to outperform on heirarchial datasets.
## Euclidean Encoder model
- Training pipeline at `scripts/train_euclidean.py`.
- Standard SentenceTransformer with Multiple Negatives Ranking loss.

## Hyperbolic Encoder model
- Training pipeline at `scripts/train_hyperbolic.py`.
- Custom Poincare ball model via Geoopt.
- Token embeddings live on a hyperbolic manifold.
 Trained using contrastive loss based on hyperbolic distance.

 ## Metrics pipeline
 - We utilised two datasets:
    - MS Marco
        - Globally recognised standard for retrieval pipelines.
    
    - WordNet
        - Heirarchial dataset, used to analyse task-specific performance on heirarchial embeddings.

- We utilised several metrics such as:
    - MRR
    - MAP
    - Recall@K
    - NDCG@K
    - Hit Rate

## Goals of the experiment
- Do hyperbolic embeddings produce better retrieval for hierarchical datasets (WordNet)?
- Does Euclidean geometry handle flat datasets like MS MARCO better?
- How do ranking metrics differ across geometries?
- Can a simple average-pooled hyperbolic encoder match SentenceTransformers?

## Results and Conclusion
In all metrics, Euclidean significantly outperforms the Hyperbolic model, by a large margin.
Likely reasons include:
- Global standard evaluation metrics are completely euclidean based.
- Hyperbolic training suffers from vanishing and exploding gradients, warped loss surfaces, etc, which are very difficult to fix in the poincare ball model.
- Improper and difficulty in implementing Manifold-aware optimizers, Mobius addition, etc with existing frameworks.
- Riemannian Optimizer is not as well developed as Euclidean optimizer, as not much work has been done on it. 