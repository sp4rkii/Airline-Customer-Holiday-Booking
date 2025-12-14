# ✈️ Airline Customer Satisfaction & Operations Assistant

## 📌 Project Overview
This project is a comprehensive solution designed to analyze airline passenger data, predict customer satisfaction, and provide an intelligent interface for operational queries. It combines **classical deep learning** for predictive analytics with a **Generative AI (Graph-RAG)** system to answer complex user questions about flights, reviews, and delays.

The system is divided into two main components:
1.  **Predictive Analysis & Modeling**: A deep learning model to forecast passenger satisfaction based on travel attributes and reviews.
2.  **AI Operations Assistant**: A hybrid Retrieval-Augmented Generation (RAG) tool using LangGraph and Neo4j to query structured and unstructured data via a Streamlit interface.

---

## 📊 Part 1: Predictive Modeling & Insights
Based on the **[Analysis Report](Analysis_Report.pdf)**, this module focuses on understanding what drives passenger satisfaction.

### 🔑 Key Findings
* **Primary Driver**: **Sentiment Score** from reviews is the #1 predictor of satisfaction, having 4x more impact than any other feature.
* **Traveler Segments**: **Business travelers** are consistently the *least* satisfied group, while First Class passengers rate highest.
* **Top Routes**: The network is heavily centered around London (LHR), with **"London to Johannesburg"** being the most popular route.
* **Pain Points**: Common complaints in negative reviews include "customer service," "flight delayed," and "seat comfort".

### 🧠 Model Performance (Model 1)
* **Architecture**: Feed-Forward Neural Network (Keras).
* **Accuracy**: **82.34%** on unseen test data.
* **F1-Score**: **80.97%**.
* **Explainability**: Validated using SHAP (Global importance) and LIME (Local instance analysis).

---

## 🤖 Part 2: Graph-RAG Assistant
The `Milestone3` directory contains an intelligent agent that allows users to query the airline database using natural language.

### 🏗️ System Architecture
The agent uses a **Hybrid Retrieval** strategy orchestrated by **LangGraph**:
* **Structured Data (Baseline)**: Converts natural language to **Cypher** queries to fetch factual data (flight counts, routes, delays) from a **Neo4j** graph database.
* **Unstructured Data (Embeddings)**: Uses **Vector Search** (FAISS) to find relevant semantic context from passenger reviews and text.
* **Synthesis**: A Large Language Model (e.g., Gemini, Mistral) combines both contexts to generate a final answer.

### ✨ Features
* **Multi-Mode Retrieval**: Choose between Baseline (Graph only), Embeddings (Vector only), or Hybrid (Both) via the UI.
* **Model Selection**: Switch between different LLMs (Gemini Flash, Mistral-7B, Zephyr-7B).
* **Transparency Layer**: View the exact Cypher queries generated and vector chunks retrieved in the "Debug" tabs.
* **Interactive UI**: Built with **Streamlit** for seamless interaction.

---

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.10+
* Neo4j Database (Local or AuraDB)
* Google Gemini API Key (or HuggingFace Hub token depending on model choice)

### 1. Clone the Repository
```bash
git clone [https://github.com/sp4rkii/airline-customer-holiday-booking.git](https://github.com/sp4rkii/airline-customer-holiday-booking.git)
cd airline-customer-holiday-booking
