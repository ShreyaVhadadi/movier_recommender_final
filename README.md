# Movie Recommender using LLM and Semantic Search

This project is a semantic movie recommendation system powered by Large Language Models (LLMs) and vector similarity search. Instead of relying on traditional genre or rating-based recommendations, this system understands the deeper meaning of user queries to suggest contextually relevant movies.

## Features

- Semantic search powered by LLM-generated embeddings
- Intelligent recommendations based on movie overviews and descriptions
- FAISS for fast similarity search on movie vectors
- Built with Python and Streamlit for a fast, interactive web interface
- Dataset of over 1 million movies (from Kaggle)

## Tech Stack

- Python
- OpenAI Embeddings (text-embedding-ada-002)
- FAISS for similarity search
- Pandas & NumPy for data processing
- Streamlit for web UI
- Kaggle dataset for movie metadata

## How It Works

1. Movie descriptions are converted into high-dimensional vectors using OpenAI's embedding model.
2. When a user enters a search prompt (e.g., "gritty sci-fi thriller about survival"), it’s embedded into the same vector space.
3. FAISS performs nearest-neighbor search to find the most semantically similar movies.
4. Results are displayed with title, year, overview, and other metadata in the Streamlit UI.

## Demo Screenshot

![Demo](path/to/your/screenshot.png)

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API Key

### Installation

```bash
git clone https://github.com/yourusername/movie-recommender-llm.git
cd movie-recommender-llm
pip install -r requirements.txt
