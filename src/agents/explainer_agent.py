# src/agents/explainer_agent.py
"""
explainer_agent.py

PURPOSE:
--------
Generates plain English explanation for loan decision.

FLOW:
-----
1. Receives decision context from orchestrator
2. Retrieves relevant knowledge via RAG (FAISS)
3. Loads top SHAP feature names
4. Sends prompt to Llama3 via Ollama
5. Returns clean explanation
"""

import sys
import os
sys.path.append(os.path.abspath("."))

from src.rag.retriever import retrieve_context
import json
import ollama
import requests


# ==============================
# 🔹 Ollama Health Check
# ==============================
def check_ollama():
    try:
        requests.get("http://localhost:11434")
        print("✅ Ollama is running")
    except:
        print("❌ Ollama not running — start it first!")
        exit()


# ==============================
# 🔹 Load SHAP Feature Names ONLY
# ==============================
def get_feature_names_only() -> list:
    """
    Returns only feature NAMES (not importance values)
    Prevents LLM from over-interpreting numbers
    """
    with open("models/top_shap_features.json", "r") as f:
        data = json.load(f)
    return list(data.keys())


# ==============================
# 🤖 Generate Explanation
# ==============================
def generate_explanation(query: str) -> str:
    """
    Parameters:
    -----------
    query : str
        Built by orchestrator — contains decision context
        Example: "Customer classified as HIGH RISK with
                  MEDIUM risk. Final decision is REVIEW."

    Returns:
    --------
    str — plain English explanation from Llama3
    """

    # Step 1: Retrieve relevant financial knowledge
    context = retrieve_context(query)

    # Step 2: Load SHAP feature names
    top_features = get_feature_names_only()

    # Step 3: Build strict prompt
    # ✅ Query contains actual decision from orchestrator
    # ✅ LLM explains THAT decision — not a hardcoded one
    prompt = f"""
You are a financial risk analyst AI assistant.

DECISION CONTEXT:
{query}

Key Features That Influenced This Decision:
{top_features}

Relevant Financial Knowledge:
{context}

STRICT RULES:
- Base your explanation ONLY on the decision context above
- DO NOT use words: high, low, stable, unstable
- DO NOT invent any customer details
- DO NOT mention LTV, mortgages, or anything not listed
- Maximum 4 sentences
- Start with: "The model's decision was based on..."
"""

    # Step 4: Call Ollama Llama3
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


# ==============================
# ▶️ MAIN — Quick Test
# ==============================
if __name__ == "__main__":
    check_ollama()
    query = ("Customer classified as HIGH RISK "
             "with MEDIUM risk level. "
             "Final decision is REVIEW.")
    result = generate_explanation(query)
    print("\n🤖 AI Financial Copilot:\n")
    print(result)