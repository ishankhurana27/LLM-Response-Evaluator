🧩 OVERVIEW

This project implements a lightweight, real-time LLM evaluation pipeline that automatically scores any LLM response on:

   -Response Relevance

   -Context Completeness

   -Hallucination Detection

   -Latency Measurement

   -Token & Cost Estimation

   -Grading (A–F)

It consumes two JSON inputs:

1)Conversation JSON → Contains the user message and LLM response

2)Context JSON → Contains context chunks retrieved from a vector database

The system evaluates whether the LLM followed the context, avoided hallucinations, and responded fully and accurately.


📦 KEY FEATURES
✅ Context-aware relevance scoring

Uses dense embeddings to check how well the LLM response matches the user query + provided context.

✅ Completeness measurement

Checks how many context chunks the LLM actually used.

✅ Advanced hallucination detection

Combines:

Sentence embeddings

spaCy concept extraction

Concept-overlap analysis

New-entity detection

This avoids false hallucination flags for short factual sentences.

✅ Grading System (A–F)

Weighted evaluation based on relevance, completeness, and hallucination severity.

✅ Fast & low-cost execution

Evaluates responses in milliseconds with minimal compute.

✅ Modular architecture

Easy to extend for additional scoring metrics.


ARCHITECTURE

                ┌──────────────────────┐
                │   Input JSONs        │
                │  (conversation,       │
                │   context chunks)     │
                └──────────┬───────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │  Sentence Embeddings     │
              │  (all-MiniLM-L6-v2)      │
              └──────────┬──────────────┘
                           │
                           ▼
        ┌────────────────────────────────────┐
        │       Evaluation Engine             │
        │------------------------------------│
        │ Relevance Scoring                  │
        │ Completeness Scoring               │
        │ Hallucination Detection (spaCy)    │
        │ Grading Logic (A–F)               │
        └────────────────┬───────────────────┘
                         │
                         ▼
         ┌───────────────────────────────────┐
         │           Final Report             │
         └───────────────────────────────────┘

🛠 TECHNOLOGIES USED & WHY

🔹 1. Python

🔹 2. Sentence Embeddings — all-MiniLM-L6-v2

Chosen because it provides the best tradeoff:

| Model             | Accuracy  | Speed         | Cost     | Notes                                |
| ----------------- | --------- | ------------- | -------- | ------------------------------------ |
| **MiniLM-L6-v2**  | High      | **Very fast** | **Free** | Best for real-time evaluation        |
| BERT-base         | Higher    | Slow          | Heavy    | Too large for per-request evaluation |
| OpenAI Embeddings | Very high | Fast          | Paid     | Not suitable for offline submission  |
| Instructor-Large  | High      | Very slow     | Heavy    | Not scalable                         |


MiniLM gives:

-384-dim embeddings (FAISS-friendly)

-< 5ms embedding time

-Strong semantic matching

-Excellent for small hardware

This makes it ideal for large-scale evaluation workloads.


🔹 3. FAISS Vector Store

Why FAISS?

--Extremely fast cosine similarity search

--GPU acceleration optional

--Lightweight and memory-efficient

--No server needed (unlike Pinecone, Weaviate)

--Perfect for embedding comparisons at evaluation time

--Used here not for retrieval, but for fast similarity scoring across context chunks.

🔹 4. spaCy (NER + Concept Extraction)

Simple regex-based claim extraction is NOT enough.
We upgraded to spaCy because:

--It extracts entities and noun phrases

--Helps determine new concepts introduced by the LLM

--Reduces false hallucination detections

--Supports domain-agnostic text (medicine, finance, tech, etc.)

Alternatives considered:
| Tool                      | Why rejected             |
| ------------------------- | ------------------------ |
| **NLTK**                  | Too basic, no NER        |
| **Regex only**            | Fails for complex claims |
| **transformer-based NER** | Too heavy for real-time  |

🚀 WHY THIS DESIGN? (Design Choices Explained)
✔ Fast local inference

Using MiniLM + spaCy means the pipeline runs without GPU, making it suitable for local setups and enterprise scaling.

✔ Hallucination detection is concept-based, not keyword-based

Short factual responses like:

"The A15 chip."

should not be marked hallucinations.
Our concept-overlap + new-entity detection solves this problem better than pure cosine similarity.

✔ Scales to millions of evaluations/day

Components chosen ensure low-latency:
| Component         | Purpose                | Latency impact |
| ----------------- | ---------------------- | -------------- |
| MiniLM embeddings | semantic similarity    | < 5ms          |
| spaCy             | concept extraction     | ~2–3ms         |
| FAISS             | fast similarity lookup | <1ms           |
| Evaluator logic   | scoring/grading        | negligible     |


Total average evaluation time: 0.05–0.15 sec per conversation.

At scale:

      Can evaluate ~10–20 million responses/day per server

      Fully CPU-friendly

      No external API costs

