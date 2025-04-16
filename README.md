# Lyra2
Lyra2: Semantic Memory AI Companion

Lyra2 is a locally-hosted AI companion with persistent, contextual memory. Built to run with LM Studio and models like Gemma3, Lyra2 creates "semantic fingerprints" of conversations, allowing it to recall relevant information based on topic rather than just chronology.
Key Features

Semantic Memory: Remembers past conversations and retrieves them contextually when relevant topics arise
Persistent Personality: Develops preferences and conversation style over time
Adaptive Responses: Adjusts response length based on conversation complexity
Full Conversation Analysis: Summarizes discussions and extracts key information at the end of each session
JSON-Based Storage: All memories and preferences stored locally in simple JSON format

Requirements

LM Studio with a compatible LLM (tested with Gemma3 17GB)
Python 3.9+ with sentence-transformers, scikit-learn, and numpy
4090 and Gemma3 17.23GB Q4

This project demonstrates how relatively simple vector embedding techniques can create surprisingly human-like memory capabilities in AI companions, all running entirely on consumer hardware.

Notes: 
