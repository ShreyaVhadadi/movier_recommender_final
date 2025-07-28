import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr
from torch.nn.functional import embedding

load_dotenv()

movies = pd.read_csv('final_dataset.csv')
base_url = "https://image.tmdb.org/t/p/w500"

movies["large_thumbnail"] = np.where(
    movies["poster_path"].isna(),
    "images.jpg",  # path to your default image
    base_url + movies["poster_path"]
)


#movies["large_thumbnail"] = movies["large_thumbnail"]+"&fife=w800"
#movies["large_thumbnail"] = np.where(
    #movies["large_thumbnail"].isna(),
    #"images.jpg",
    #movies["large_thumbnail"]
#)

raw_documents2 = TextLoader("movies_tagged_overview1.txt").load()
text_splitter1 = CharacterTextSplitter(chunk_size=500, chunk_overlap=0, separator="\n")
documents2 = text_splitter1.split_documents(raw_documents2)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db_movies = Chroma.from_documents(
    documents2,
    embedding=embeddings)

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    recs = db_movies.similarity_search(query, k=initial_top_k)
    movies_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    movies_recs = movies[movies["id"].isin(movies_list)]

    if category != "All":
        movies_recs = movies_recs[movies_recs["simple_categories"] == category]

    if tone == "Happy":
        movies_recs = movies_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        movies_recs = movies_recs.sort_values(by="surprise", ascending=True)
    elif tone == "Angry":
        movies_recs = movies_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        movies_recs = movies_recs.sort_values(by="fear", ascending=True)
    elif tone == "Sad":
        movies_recs = movies_recs.sort_values(by="sadness", ascending=False)

    return movies_recs.head(final_top_k)

def wrap_text_words(text, width=40):
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 <= width:
            current_line += (" " if current_line else "") + word
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return '\n'.join(lines)

def recommend_movies(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []
    for _, row in recommendations.iterrows():
        description = row["overview"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:24]) + "..."
        wrapped_description = wrap_text_words(truncated_description, width=40)
        caption = f"{row['title']}:\n{wrapped_description}"
        #caption = f"{row['title']}:\n{truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

categories = ["All"] + sorted(movies["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("## Movie Recommendations")

    with gr.Row():
        user_query = gr.Textbox(label="Please enter a description of a movie",
                                placeholder="e.g., A movie about love")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select a tone", value="All")
        submit_button = gr.Button(value="Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommendations",columns=4,rows=4)

    submit_button.click(
        recommend_movies,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )

if __name__ == "__main__":
    dashboard.launch()
