# Cerebrix 🧠🤖

[![Python Version](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Framework](https://img.shields.io/badge/Framework-LangGraph-blueviolet)](https://langchain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Your AI teaching assistant for YouTube courses, combining video lectures, textbooks, and Socratic tutoring in one conversational interface.

<img width="1899" height="1045" alt="image" src="https://github.com/user-attachments/assets/ac805f4c-96a6-4dd9-a65d-b9bd51584bfb" />

<img width="1914" height="1041" alt="image" src="https://github.com/user-attachments/assets/949e91ce-d789-4ed4-b4f7-db40bb0aaf78" />


---

## 📖 Description

Cerebrix transforms passive video learning into an interactive educational experience. This AI-powered teaching assistant can answer questions, explain concepts, and provide Socratic guidance for any YouTube course with accompanying materials.

The current deployment focuses on Stanford's CS109 (Probability for Computer Scientists), but the architecture is course-agnostic and can be adapted to any subject.

**Why Cerebrix?**

*   **Fills the Learning Gap:** Transforms one-way video lectures into interactive conversations.
*   **Learn By Doing:** Receive guided hints on problems rather than direct answers.
*   **Course-Agnostic:** Adaptable architecture supports any YouTube course.
*   **Always Available:** 24/7 access to course-specific tutoring without waiting for office hours.
*   **Multi-Modal Understanding:** Processes both text and images for comprehensive support.

---

## 💡 Motivation

While studying Stanford's CS109 on YouTube, I encountered a common frustration: there was no way to discuss problems or get guidance when attempting exercises. As someone passionate about understanding the mathematics behind AI, I wanted a TA or study companion to support this journey, so I built one.

This project emerged from that need, creating an intelligent assistant that any self-directed learner can use to enhance their online education experience.

---


## Features

-   **Multi-Modal Knowledge Base:** Ingests and understands content from both a PDF textbook and a full YouTube video playlist.
-   **Advanced RAG Pipeline:** Uses a sophisticated Retrieval-Augmented Generation (RAG) pipeline with metadata filtering for precise, context-aware answers.
-   **Socratic Tutoring:** When a user submits their work (via text or image), the agent analyzes their approach and provides guiding hints rather than direct answers.
-   **Conversational Memory:** Remembers the context of the conversation for follow-up questions.
-   **Powered by LangGraph:** Built on a robust, stateful agent architecture using LangGraph.

## Project Architecture

-   **Backend:** A `FastAPI` server running a `LangGraph` agent.
-   **Frontend:** A static `Nextjs` chat interface.
-   **Vector Database:** `AstraDB` for storing and retrieving text and metadata embeddings.
-   **Database:** `PostgreSQL` for storing chat history.
-   **Image Hosting:** `Cloudinary` for storing page images and video keyframes.
-   **LLMs:** Google `Gemini` for reasoning, vision, and generation.
-   **Data Ingestion:** Custom Python scripts using `PyMuPDF`, `OpenCV`, and `yt-dlp` to process the knowledge sources.

---

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

This project uses `conda` to manage its complex dependencies.

```bash
# Clone the repository
git clone https://github.com/ChinmayNakwa/Cerebrix.git
cd Cerebrix

# Create and activate the conda environment
conda create --name cerebrix python=3.12
conda activate cerebrix

# Install system-level dependencies using conda
conda install -c conda-forge --file conda-reqs.txt  

# Or install them one by one:
conda install -c conda-forge poppler ffmpeg opencv

# Install Python packages using pip
pip install -r requirements.txt
