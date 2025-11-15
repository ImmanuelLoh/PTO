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
    def __init__(self, index_path, metadata_path, embedding_model="text-embedding-3-small"):
        self.index = faiss.read_index(index_path)

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.client = OpenAI()
        self.embedding_model = embedding_model

    def query(self, query_texts, n_results=5):
        query = query_texts[0]

        # Create query embedding
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=query
        )

        q = np.array(response.data[0].embedding).astype("float32")

        # Normalize for cosine similarity
        faiss.normalize_L2(q.reshape(1, -1))

        # Perform FAISS search
        distances, indices = self.index.search(q.reshape(1, -1), n_results)

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
        
        # Create FAISS Retrievers for tables and text
        self.table_collection = FAISSRetriever(
            index_path=faiss_index_path,
            metadata_path=metadata_json_path,
            embedding_model="text-embedding-3-small"
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
        
        result = {
            "query": user_query,
            "answer": answer,
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
            collection_type = search['collection']
            
            # Select collection
            collection = self.table_collection if collection_type == "tables" else None

            if collection is None:
                if verbose:
                    print(f"Skipping unknown collection type: {collection_type}")
                continue
            
            # Perform search
            results = collection.query(
                query_texts=[query_text],
                n_results=5  # Top 5 per search
            )
            
            if verbose:
                print(f"Search {i}: '{query_text}' in {collection_type}")
                print(f"  Retrieved: {len(results['ids'][0])} chunks")
            
            # Convert to RetrievedChunk objects
            for j in range(len(results['ids'][0])):
                chunk = RetrievedChunk(
                    content=results['documents'][0][j],
                    source=results['metadatas'][0][j].get('source', 'unknown'),
                    score=results['distances'][0][j] if 'distances' in results else 0.0,
                    metadata=results['metadatas'][0][j]
                )
                all_chunks.append(chunk)
        
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