"""
Agentic RAG System for Financial Document Analysis
Multi-stage retrieval with planning and execution phases
"""

import json
import faiss
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class RetrievedChunk:
    """Container for retrieved document chunks"""
    content: str
    source: str
    score: float
    metadata: Dict[str, Any]

class FAISSRetriever:
    def __init__(self, index_path, metadata_path):
        self.index = faiss.read_index(index_path)

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def search(self, embedding, n_results=5):
        vec = embedding.astype("float32")
        faiss.normalize_L2(vec.reshape(1, -1))  # Normalize for cosine similarity

        # Perform FAISS search
        distances, indices = self.index.search(vec.reshape(1, -1), n_results)

        docs, metas = [], []
        for idx in indices[0]:
            metas.append(self.metadata[idx]["metadata"])
            docs.append(self.metadata[idx]["content"])

        return {
            "ids": indices.tolist(),
            "distances": distances.tolist(),
            "documents": [docs],
            "metadatas": [metas]
        }


class TableAgenticRAG:
    """
    Two-stage RAG system:
    1. Planning: LLM identifies what data is needed
    2. Retrieval: Multiple targeted searches
    3. Synthesis: Answer with complete context
    """
    
    # Initialize with Faiss index and OpenAI client
    def __init__(self, faiss_index_path: str, metadata_json_path: str):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        with open(metadata_json_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # FAISS Retrievers for tables
        self.faiss_retriever = FAISSRetriever(
            index_path=faiss_index_path,
            metadata_path=metadata_json_path,
        )
        
        # Model configuration
        self.planning_model = "gpt-4o-mini"  # Fast for planning
        self.synthesis_model = "gpt-4o"      # Better for calculations
        
    def query(self, user_query: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Main query interface with timing instrumentation
        """

        if verbose:
            print(f"\n{'='*60}")
            print(f"QUERY: {user_query}")
            print(f"{'='*60}\n")
        
        # Stage 1: Planning
        retrieval_plan = self._create_retrieval_plan(user_query, verbose)
        
        # Stage 2: Multi-retrieval
        all_chunks = self._execute_retrieval_plan(retrieval_plan, verbose)
        
        # Stage 3: Synthesis
        answer = self._synthesize_answer(user_query, all_chunks, verbose)
                
        # Extract sources
        sources = list(set([chunk.source for chunk in all_chunks]))

        chunk_results = [
            {
                "content": chunk.content,
                "score": chunk.score,
                "metadata": chunk.metadata
            }
            for chunk in all_chunks
        ]

        result = {
            "query": user_query,
            "answer": answer,
            "results": chunk_results,
            "sources": sources,
            "metadata": {
                "num_retrievals": len(retrieval_plan.get("searches", [])),
                "num_chunks": len(all_chunks)
            }
        }
        return result
    
    def _create_retrieval_plan(self, query: str, verbose: bool) -> Dict:
        """
        Stage 1: LLM creates a retrieval plan
        Identifies what specific data points are needed
        """
        
        planning_prompt = f"""You are a financial analyst planning data retrieval for a query.

USER QUERY: {query}

IMPORTANT CONTEXT:
- The current year is 2025.
- Financial data is organized in two main collections: tables and text documents.

Analyze this query and determine:
1. What specific financial metrics are needed? (e.g., "Revenue Q2 2025", "Operating Expenses 2023")
2. What time periods? (quarters/years)
3. Should we search tables or text documents?

AVAILABLE COLLECTIONS:
- tables: Financial statements, income statements, balance sheets (structured data)
- text: Earnings call transcripts, MD&A sections (narrative explanations)

OUTPUT FORMAT (strict JSON):
{{
    "query_type": "calculation|comparison|trend|explanation",
    "metrics_needed": [
        "specific metric name and period"
    ],
    "searches": [
        {{
            "search_query": "concise search string",
            "collection": "tables|text",
            "reason": "why this search is needed"
        }}
    ]
}}

Think step-by-step about what data is required to answer the query completely.
Output ONLY the JSON, no other text."""

        if verbose:
            print("PLANNING PHASE")
            print("-" * 60)
        
        response = self.client.chat.completions.create(
            model=self.planning_model,
            messages=[{"role": "user", "content": planning_prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        plan = json.loads(response.choices[0].message.content)
        
        if verbose:
            print(f"Query Type: {plan.get('query_type', 'unknown')}")
            print(f"Metrics Needed: {', '.join(plan.get('metrics_needed', []))}")
            print(f"Planned Searches: {len(plan.get('searches', []))}")
            for i, search in enumerate(plan.get('searches', []), 1):
                print(f"  {i}. [{search['collection']}] {search['search_query']}")
                print(f"     Reason: {search['reason']}")
            print()
        
        return plan
    
    def _execute_retrieval_plan(self, plan: Dict, verbose: bool) -> List[RetrievedChunk]:
        """
        Stage 2: Execute multiple targeted retrievals
        """
        
        if verbose:
            print("RETRIEVAL PHASE")
            print("-" * 60)
        
        all_chunks = []
        
        for i, search in enumerate(plan.get('searches', []), 1):
            query_text = search['search_query']

            # Embedding for the search query
            emb = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=query_text
            )
            search_embedding = np.array(emb.data[0].embedding).astype("float32")

            # Pass embedding to FAISS retriever
            results = self.faiss_retriever.search(
                embedding=search_embedding
            )

            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]

            for j in range(len(docs)):
                all_chunks.append(
                    RetrievedChunk(
                        content=docs[j],
                        source=metas[j].get("source", "unknown"),
                        score=distances[j],
                        metadata=metas[j]
                    )
                )
            
        # Remove duplicates based on content
        unique_chunks = []
        seen_content = set()

        for chunk in all_chunks:
            content_hash = hash(chunk.content[:200])  # Hash first 200 chars
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_chunks.append(chunk)
        
        if verbose:
            print(f"\nTotal chunks retrieved: {len(all_chunks)}")
            print(f"Unique chunks: {len(unique_chunks)}")
            print()
        
        return unique_chunks
    
    def _synthesize_answer(self, query: str, chunks: List[RetrievedChunk], verbose: bool) -> str:
        """
        Stage 3: Generate final answer with all retrieved context
        """
        
        if verbose:
            print("GENERATE ANSWER PHASE")
            print("-" * 60)
        
        # Format context
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[Chunk {i} - {chunk.source}]\n{chunk.content}\n")
        
        context = "\n".join(context_parts)
        
        synthesis_prompt = f"""You are a financial analyst providing precise answers with calculations.

USER QUERY: {query}

RETRIEVED DATA:
{context}

INSTRUCTIONS:
1. Answer the query using ONLY the data provided above
2. If calculations are needed, show ALL steps clearly
3. Present data in tables where appropriate
4. If data is missing, state what's missing specifically
5. Cite sources using the format: (Source: document-name)
6. For trends/comparisons, explain the implications briefly

Be precise, show your work, and format your answer professionally."""

        response = self.client.chat.completions.create(
            model=self.synthesis_model,
            messages=[{"role": "user", "content": synthesis_prompt}],
            temperature=0.1
        )
        
        answer = response.choices[0].message.content
        
        if verbose:
            print(f"Answer generated: {len(answer)} characters")
            print()
        
        return answer