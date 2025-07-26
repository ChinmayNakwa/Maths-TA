# Maths AI Tutor 🧠🤖

[![Python Version](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Framework](https://img.shields.io/badge/Framework-LangGraph-blueviolet)](https://langchain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced, multi-modal AI-powered teaching assistant for the Probability. This agent provides a conversational interface to answer questions, explain concepts from the course textbook and video lectures, and offer Socratic-style guidance on user-submitted problems.

## Features

-   **Multi-Modal Knowledge Base:** Ingests and understands content from both a PDF textbook and a full YouTube video playlist.
-   **Advanced RAG Pipeline:** Uses a sophisticated Retrieval-Augmented Generation (RAG) pipeline with metadata filtering for precise, context-aware answers.
-   **Socratic Tutoring:** When a user submits their work (via text or image), the agent analyzes their approach and provides guiding hints rather than direct answers.
-   **Conversational Memory:** Remembers the context of the conversation for follow-up questions.
-   **Powered by LangGraph:** Built on a robust, stateful agent architecture using LangGraph.

## Project Architecture

-   **Backend:** A `FastAPI` server running a `LangGraph` agent.
-   **Frontend:** A static `HTML/CSS/JS` chat interface.
-   **Vector Database:** `AstraDB` for storing and retrieving text and metadata embeddings.
-   **Image Hosting:** `Cloudinary` for storing page images and video keyframes.
-   **LLMs:** Google `Gemini 2.5 Flash` for reasoning, vision, and generation.
-   **Data Ingestion:** Custom Python scripts using `PyMuPDF`, `OpenCV`, and `yt-dlp` to process the knowledge sources.

---
<img width="1016" height="1063" alt="Screenshot 2025-07-24 003015" src="https://github.com/user-attachments/assets/a68c5478-f07c-40d0-b917-b2c7a74bf736" />

<img width="1141" height="705" alt="flow" src="https://github.com/user-attachments/assets/ef90fc40-d290-4ff4-9a68-9b67d89a3624" />

## 🚀 Getting Started

Follow these steps to set up and run the Maths AI Tutor on your local machine.

### 1. Prerequisites

-   [Anaconda or Miniconda](https://www.anaconda.com/download) installed.
-   Access keys for:
    -   Google AI (Gemini)
    -   AstraDB
    -   Cloudinary
    -   LangSmith (optional, but highly recommended for debugging)

### 2. Environment Setup

This project uses `conda` to m[Unit I.pptx](https://github.com/user-attachments/files/21442390/Unit.I.pptx)anage its complex dependencies.

```bash
# Clone the repository
git clone https://github.com/your-username/Maths-TA.git
cd Maths-TA

# Create and activate the conda environment
conda create --name maths-ta python=3.12
conda activate maths-ta

# Install system-level dependencies using conda
conda install -c conda-forge --file conda-reqs.txt # You would create this file
# Or install them one by one:
conda install -c conda-forge poppler ffmpeg opencv

# Install Python packages using pip
pip install -r requirements.txt
