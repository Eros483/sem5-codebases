from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import csv
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger=get_logger(__name__)

def load_train_examples(query_path="wordNet/queries.tsv", corpus_path="wordNet/corpus.tsv"):
    """
    Load training examples from the given query and corpus files.

    Args:
        query_path (str): Path to the TSV file containing queries. 
        corpus_path (str): Path to the TSV file containing corpus passages.
    
    Returns:
        List: A list of examples for training.
    """
    try:
        corpus={}
        with open(corpus_path, "r") as fc:
            for row in csv.reader(fc, delimiter='\t'):
                corpus[row[0]]=row[1]
        
        examples=[]
        with open(query_path, "r") as fq:
            for row in csv.reader(fq, delimiter='\t'):
                _, qtext, pos_id=row
                pos_id=int(pos_id)
                examples.append(InputExample(texts=[qtext, corpus[str(pos_id)]]))
        logger.info(f"Loaded {len(examples)} training examples.")
        return examples

    except Exception as e:
        logger.error(f"Error loading training examples: {e}")
        raise CustomException("Error loading training examples", e)

def train_euclidean_model():
    """
    Train a dual embedding model using Euclidean distance.
    """
    try:
        model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        examples=load_train_examples()
        dataloader=DataLoader(examples, shuffle=True, batch_size=32)
        train_loss=losses.MultipleNegativesRankingLoss(model)

        model.fit(
            train_objectives=[(dataloader, train_loss)],
            epochs=5,
            warmup_steps=100,
            use_amp=True,
            show_progress_bar=True
        )

        model.save("models/wordNet/euclidean/")
        logger.info("Model trained and saved.")
    
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise CustomException("Error during model training", e)

if __name__ == "__main__":
    train_euclidean_model()