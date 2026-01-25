# Cerebrix 🧠🤖

Your AI teaching assistant for YouTube courses, combining video lectures, textbooks, and Socratic tutoring in one conversational interface.

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

## ✨ Features

*   **Multi-Modal Knowledge Base:** Ingests and understands content from both PDF textbooks and full YouTube video playlists.
*   **Advanced RAG Pipeline:** Uses sophisticated Retrieval-Augmented Generation with metadata filtering for precise, context-aware answers.
*   **Socratic Tutoring:** Analyzes user-submitted work (text or image) and provides guiding hints rather than direct solutions.
*   **Conversational Memory:** Maintains conversation context for natural follow-up questions.
*   **Powered by LangGraph:** Built on a robust, stateful agent architecture for reliable performance.

---

## 🎯 Use Cases

*   **Self-Paced Learners:** Get immediate help while studying MOOCs and online courses.
*   **Homework Guidance:** Receive Socratic hints that guide your thinking without giving away solutions.
*   **Concept Review:** Ask follow-up questions about lecture content in natural language.
*   **Visual Problem Solving:** Upload images of your work for AI-guided feedback.

---

## 🏗️ Project Architecture

*   **Backend:** FastAPI server running a LangGraph agent.
*   **Frontend:** Static HTML/CSS/JS chat interface.
*   **Vector Database:** AstraDB for storing and retrieving text and metadata embeddings.
*   **Image Hosting:** Cloudinary for storing page images and video keyframes.
*   **LLMs:** Google Gemini 2.5 Flash for reasoning, vision, and generation.
*   **Data Ingestion:** Custom Python scripts using PyMuPDF, OpenCV, and yt-dlp to process knowledge sources.

---

<img width="1141" height="705" alt="System Architecture Flow" src="https://github.com/user-attachments/assets/ef90fc40-d290-4ff4-9a68-9b67d89a3624" />

---

## 🚀 Getting Started

Follow these steps to set up and run Cerebrix on your local machine.

### 1. Prerequisites

*   Anaconda or Miniconda installed
*   Python 3.12
*   API keys for:
    *   Google AI (Gemini)
    *   AstraDB
    *   Cloudinary
    *   LangSmith (optional, but recommended for debugging)

### 2. Clone the Repository

```bash
git clone https://github.com/ChinmayNakwa/Cerebrix.git
cd Cerebrix
git checkout main
```

### 3. Environment Setup

This project uses conda to manage complex dependencies.

```bash
# Create and activate the conda environment
conda create --name cerebrix python=3.12
conda activate cerebrix

# Install system-level dependencies using conda
conda install -c conda-forge poppler ffmpeg opencv

# Install Python packages using pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory with your API keys:

```dotenv
GOOGLE_API_KEY=
ASTRA_DB_API_ENDPOINT=
ASTRA_DB_APPLICATION_TOKEN=
ASTRA_DB_KEYSPACE=
ASTRA_DB_COLLECTION_NAME=
PLAYLIST_URL=
PDF_PATH=
CLOUDINARY_CLOUD_NAME=
CLOUDINARY_API_KEY=
CLOUDINARY_API_SECRET=
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=
HF_TOKEN=
```

> **⚠️ Note:** Video processing can take several hours depending on playlist length. The system extracts keyframes, transcripts, and metadata for each video.

### 5. Run the Application

```bash
# Start the FastAPI backend
uvicorn app.main:app --reload
```


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

---

## 🙏 Acknowledgments

*   Stanford CS109 course materials
*   LangChain and LangGraph teams
*   Google Gemini AI
*   The open-source community

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub or reach out to `29chinmaynakwa@gmail.com`.

---

> **⚠️ Important Note:** When replicating this project for other courses, be mindful of copyright and fair use policies for course materials. Ensure you have appropriate permissions to process and use the content.

## ⚖️ Copyright & Fair Use Notice

This project is designed as an educational tool to enhance learning from publicly available course materials. We believe this constitutes fair use under copyright law for the following reasons:

-   **Educational Purpose**: Non-commercial teaching aid for self-directed learners
-   **Transformative Use**: Creates an interactive tutoring interface not present in original materials
-   **Attribution**: All content is attributed to original creators with links to source materials
-   **No Market Harm**: Designed to complement, not replace, the original course experience

**All course content remains the property of its original creators.** This project:

-   Does not claim ownership of any course materials
-   Is not monetized in any way
-   Can be removed upon request from content owners

If you are a content owner and have concerns about this project's use of your materials, please contact `29chinmaynakwa@gmail.com` and we will respond promptly.

**For other users replicating this project**: Please ensure you have appropriate permissions for any copyrighted materials you process. Fair use determinations are made on a case-by-case basis# Cerebrix 🧠🤖

Your AI teaching assistant for YouTube courses, combining video lectures, textbooks, and Socratic tutoring in one conversational interface.

---
