🧩 Overview

This project implements a lightweight, real-time LLM evaluation pipeline that automatically scores any LLM response on:

Response Relevance

Context Completeness

Hallucination Detection

Latency Measurement

Token & Cost Estimation

Grading (A–F)

It consumes two JSON inputs:

Conversation JSON → Contains the user message and LLM response

Context JSON → Contains context chunks retrieved from a vector database

The system evaluates whether the LLM followed the context, avoided hallucinations, and responded fully and accurately.
