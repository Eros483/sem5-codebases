from datasets import load_dataset
import csv
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger=get_logger(__name__)

def prepare_MS_Marco_Subset(limit=1000):
    """
    Prepare a subset of the MS MARCO dataset for testing purposes.

    Args:
        limit (int): The number of samples to include in the subset.
    """
    try:
        ds=load_dataset("ms_marco", "v2.1", split="train[:1%]")

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise CustomException("Failed to load dataset", e)

    # print("Available fields:", ds[0].keys())
    # print("Sample item:", ds[0])

    try:
        with open("ms_marco/corpus.tsv", "w", newline='') as fc, open("ms_marco/queries.tsv", "w", newline='') as fq:
            corpus_writer = csv.writer(fc, delimiter='\t')
            query_writer = csv.writer(fq, delimiter='\t')

            for i, item in enumerate(ds):
                if i >= limit:
                    break

                if item['passages'] and item['passages']['passage_text']:
                    passage_text=item['passages']['passage_text'][0]
                    corpus_writer.writerow([i, passage_text])

                try:
                    query_writer.writerow([i, item["query"], i])

                except Exception as e:
                    logger.error(f"Failed to write query for item {i}: {e}")
                    logger.info(f"Available fields: {ds[0].keys()}")
                    raise CustomException(f"Failed to write query for item {i}", e)
                
        logger.info(f"Prepared MS MARCO subset with {limit} samples.")

    except Exception as e:
        logger.error(f"Failed to write data files: {e}")
        raise CustomException("Failed to write data files", e)
    
if __name__ == "__main__":
    prepare_MS_Marco_Subset(limit=1000)