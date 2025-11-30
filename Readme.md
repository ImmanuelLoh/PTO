# CFO Agent

## 📖 Project Overview
The CFO Agent is an advanced Retrieval-Augmented Generation (RAG) system designed to answer complex financial questions with the reasoning capabilities of a Chief Financial Officer.

Driven by the central orchestrator Agent_CFO_Project.ipynb, this system does not simply "lookup" facts; it employs an autonomous agent that plans its investigation, gathers evidence from multiple modalities (text, tables, and images), and synthesizes a citation-backed answer.

The current knowledge base consists of Alphabet Inc. (Google) financial filings (10-K, 10-Q) and investor presentation decks.

The baseline version can be found at baseline.ipynb

## ⚙️ Architecture & Pipeline 

### 1. The Data Strategy (Google Financials)
Unlike standard RAG systems that treat all data as flat text, this pipeline segregates data by structure to maximize retrieval accuracy:
* **Text Corpus:** Narrative sections (MD&A, Footnotes, Risk Factors) are chunked and embedded for semantic search.
* **Tabular Store:** Financial statements (Balance Sheets, Income Statements) are processed to preserve row/column relationships.
* **Visual Store:** Presentation slides are processed via OCR to capture visual data (charts, graphs) often missed by text parsers.

### 2. Execution Workflow
The system follows a high-performance **Async Parallel Pipeline**

#### Flow Description
1. User Query enters the system.  
2. System checks the Semantic Cache.  
    - If Hit: Returns cached answer immediately.  
    - If Miss: Initiates Parallel Retrieval Engine.  
3. The Async Retrieval Layer fetches data simultaneously from:
    - Text Vector Store  
    - Table Store
    - Image/OCR Store
4. All data is merged into a Unified Context Window.
5. The CFO Agent receives the context and begins reasoning (see below).  
6. Agent generates a Final Answer with citations and updates the Semantic Cache.

## HOW THE AGENT WORKS

The Agent is governed by a specialized Chain-of-Thought (CoT) system prompt that enforces a strict 3-phase reasoning process. This ensures the model behaves like a financial analyst rather than a generic chatbot.  

### Phase 1  
Planning Before calling any tools, the agent analyzes the user's request (e.g., "What was the YoY growth in Cloud revenue?") and determines which data modalities are required. (Internal Thought: "I need structured numbers for revenue (Table) and potentially narrative explanations for the growth drivers (Text).")

### Phase 2 Pruning (Plan Refinement)
The agent reviews the potential utility of the tools. If the query is purely qualitative, it may prune the "Table" retrieval to reduce noise. If the query is purely numeric, it focuses strictly on the "Table" tool.

### Phase 3 Execution & Synthesis

Retrieval: The system searches vector stores populated with Google's financial chunks using embedding similarity.  
Computation: The agent performs explicit numerical computations (calculating margins, deltas) based on retrieved data.  
Citation: It merges the evidence into a final response, strictly citing the source documents (e.g., Source: 2024 10-K, Table 4).

## BENCHMARKING & EVALUATION

To ensure the system meets production standards, we implemented a rigorous benchmarking framework comparing three distinct configurations.

### Test Scenarios 
We evaluate performance across three system states
    - *Baseline*: Rule Base Retrieval with Naive Chunking  
    - *Optimized (Current)*: Async parallel multi-modal retrieval + Agentic "Plan-Prune-Execute" workflow.  
    - *Cached*: Optimized agent + Semantic Caching layer enabled.

### Performance Metrics (Timing)  
We instrumented the code to log detailed latency breakdowns for every query. We track p50 (Median) and p95 (Tail Latency) for:  

    - Total Latency: End-to-end time from user query to final answer.  
    - Retrieval Time: Time taken to fetch data from Vector Stores (Text/Table/Image).  
    - Reasoning Time: Time spent by the Agent in the Planning and Pruning phases.  
    - Generation Time: Time spent generating the final prose response.  

### RAG Quality Metrics  
The "RAG Triad" is tested through the 3 Metrics:

    - Context Precision: Is the retrieved evidence relevant to the query?  
    - Faithfulness: Is the answer supported by the retrieved contexts?  
    - Answer Relevance: Does the final answer actually address the user's specific question?
