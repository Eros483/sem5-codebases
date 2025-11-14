import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
from tqdm import tqdm
import csv
import os
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from utils.custom_exception import CustomException
from utils.logger import get_logger

logger=get_logger(__name__)

class TextDataset(Dataset):
    """
    Custom Dataset for loading text data from queries and corpus files.
    """
    def __init__(self, queries_path, corpus_path, tokenizer):
        self.tokenizer=tokenizer
        self.samples=[]
        corpus={}

        with open(corpus_path, "r") as fc:
            for row in csv.reader(fc, delimiter='\t'):
                corpus[row[0]]=row[1]
        
        with open(queries_path, "r") as fq:
            for row in csv.reader(fq, delimiter='\t'):
                _, qtext, pos_id=row
                pos_id=int(pos_id)
                self.samples.append((qtext, corpus[str(pos_id)]))
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        q, d=self.samples[idx]
        q_tok=self.tokenizer(q, return_tensors='pt', padding='max_length', truncation=True, max_length=32)
        d_tok=self.tokenizer(d, return_tensors='pt', padding='max_length', truncation=True, max_length=32)
        return q_tok["input_ids"].squeeze(0), d_tok["input_ids"].squeeze(0)

class HyperbolicDualEncoder(nn.Module):
    """
    Hyperbolic Encoder Model using poincare ball geometry.
    """
    def __init__(self, vocab_size, dim=128, c=1.0):
        super().__init__()
        self.manifold=geoopt.PoincareBall(c=c)
        self.emb=geoopt.ManifoldParameter(self.manifold.random_normal((vocab_size, dim)), manifold=self.manifold)
    
    def forward(self, input_ids):
        token_vecs=self.emb[input_ids]
        tang=self.manifold.logmap0(token_vecs)
        mean_tang=tang.mean(dim=1)
        sent=self.manifold.expmap0(mean_tang)
        return self.manifold.projx(sent)
    
def train_model(model, loader, optimizer, manifold, device, temp=0.07):
    """
    Train the hyperbolic encoder model for one epoch.
    """
    model.train()
    for batch in tqdm(loader):
        q, d=[x.to(device) for x in batch]
        q_emb=model(q)
        d_emb=model(d)

        dist=manifold.dist(q_emb.unsqueeze(1), d_emb.unsqueeze(0))
        logits=-dist/temp

        labels=torch.arange(logits.size(0), device=device)
        loss=F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for p in model.parameters():
            if hasattr(p, 'manifold'):
                p.data=p.manifold.projx(p.data)
    return loss.item()

def train_hyperbolic_model():
    try:
        tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        vocab_size= tokenizer.vocab_size
        logger.info("Tokenizer loaded.")

    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        raise CustomException("Error loading tokenizer", e)

    try:
        dataset=TextDataset("wordNet/queries.tsv", "wordNet/corpus.tsv", tokenizer)
        dataloader=DataLoader(dataset, batch_size=16, shuffle=True)
        logger.info(f"Dataset loaded with {len(dataset)} samples.")
    
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise CustomException("Error loading dataset", e)

    device="cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    model=HyperbolicDualEncoder(vocab_size=vocab_size, dim=128).to(device)
    optimizer=geoopt.optim.RiemannianSGD(model.parameters(), lr=1e-3)

    try:
        for epoch in range(5):
            loss=train_model(model, dataloader, optimizer, model.manifold, device)
            logger.info(f"Epoch {epoch+1}/5, Loss: {loss:.4f}")
        
        os.makedirs("models/wordNet/hyperbolic", exist_ok=True)
        torch.save(model.state_dict(), "models/wordNet/hyperbolic/model.pth")
        logger.info("Model trained and saved.")

    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise CustomException("Error during model training", e)

if __name__ == "__main__":
    train_hyperbolic_model()