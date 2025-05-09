from typing import List, Dict, Optional
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from llama_index.core import Document
from src.utils.logger import Logger

class RAGSystem:
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        k: int = 4,
        logger: Optional[Logger] = None
    ):
        self.embedding_model = embedding_model
        self.k = k
        self.logger = logger or Logger()
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.ensemble_retriever = None
        self.original_documents = None

    def setup_retrievers(self, documents: List[Document]) -> None:
        """Set up sparse (BM25), dense (FAISS), and ensemble retrievers."""
        # Store original documents
        self.original_documents = documents
        
        # Extract texts for retrievers
        texts = []
        for doc in documents:
            try:
                if hasattr(doc, 'text'):
                    texts.append(doc.text)
                elif hasattr(doc, 'get_content'):
                    texts.append(doc.get_content())
                elif hasattr(doc, 'page_content'):
                    texts.append(doc.page_content)
                else:
                    self.logger.warning(f"Document has no accessible text content: {doc}")
                    continue
            except Exception as e:
                self.logger.error(f"Error extracting text from document: {str(e)}")
                continue
        
        if not texts:
            raise ValueError("No valid texts could be extracted from documents")
        
        # BM25 retriever
        bm25_retriever = BM25Retriever.from_texts(
            texts,
            k=self.k, 
            b=0.75,
            k1=1.5
        )
        
        # FAISS vector store and retriever
        embedding = HuggingFaceEmbeddings(model_name=self.embedding_model)
        faiss_vectorstore = FAISS.from_texts(texts, embedding)
        faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": self.k}) 
        
        # ensemble retriever with weights
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.3, 0.7],  # weights to favor dense retrieval
            top_k_limit=self.k
        )
        
        self.logger.info("Retrievers setup completed")

    def retrieve_documents(self, query: str) -> List[Document]:
        """Retrieve documents using the ensemble retriever."""
        if not self.ensemble_retriever:
            raise ValueError("Retrievers not set up. Call setup_retrievers first.")
        
        self.logger.info("Retrieving documents using ensemble method...")
        try:
            # Get raw results from ensemble retriever
            raw_results = self.ensemble_retriever.invoke(query)
            
            if not raw_results:
                self.logger.warning("No documents retrieved for the query. Returning empty list.")
                return []
            
            # Convert raw results to text if needed
            retrieved_texts = []
            for result in raw_results:
                if isinstance(result, str):
                    retrieved_texts.append(result)
                elif hasattr(result, 'text'):
                    retrieved_texts.append(result.text)
                elif hasattr(result, 'page_content'):
                    retrieved_texts.append(result.page_content)
                else:
                    self.logger.warning(f"Unexpected result type: {type(result)}")
                    continue
            
            # Find matching documents from original documents
            retrieved_docs = []
            for text in retrieved_texts:
                found_match = False
                for doc in self.original_documents:
                    try:
                        doc_text = doc.text if hasattr(doc, 'text') else (
                            doc.get_content() if hasattr(doc, 'get_content') else doc.page_content
                        )
                        if doc_text == text:
                            retrieved_docs.append(doc)
                            found_match = True
                            break
                    except Exception as e:
                        self.logger.error(f"Error comparing document text: {str(e)}")
                        continue
                
                if not found_match:
                    self.logger.warning(f"Could not find matching document for retrieved text: {text[:100]}...")
            
            if not retrieved_docs:
                self.logger.warning("No matching documents found in original documents.")
            
            return retrieved_docs
            
        except Exception as e:
            self.logger.error(f"Error during document retrieval: {str(e)}")
            raise

    def rerank_documents(self, query: str, docs: List[Document], top_k: int = 4) -> List[Document]:
        """Rerank documents using Cross-Encoder."""
        self.logger.info("Reranking documents...")
        
        # pairs of (query, document) for scoring
        pairs = [[query, doc.get_content() if hasattr(doc, 'get_content') else doc.page_content] for doc in docs]
        
        # get scores for each pair
        scores = self.reranker.predict(pairs)
        
        # sort documents by score
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # return top_k documents
        return [doc for doc, score in scored_docs[:top_k]] 