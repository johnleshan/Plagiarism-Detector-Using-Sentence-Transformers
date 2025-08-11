<p align="center">
  <img src="https://via.placeholder.com/150x60?text=PlagiarismDetector" alt="Logo">
</p>

<h1 align="center">Plagiarism Detector Using Sentence Transformers</h1>

<p align="center">
  Detect semantic plagiarism using AI-powered sentence embeddings.
  <br>
  <br>
  <a href="https://github.com/johnleshan/Plagiarism-Detector-Using-Sentence-Transformers/issues">Report Bug</a>
  ·
  <a href="https://github.com/johnleshan/Plagiarism-Detector-Using-Sentence-Transformers/issues">Request Feature</a>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Made with Sentence Transformers](https://img.shields.io/badge/Powered%20by-Sentence%20Transformers-green)](https://www.sbert.net/)

## About The Project

Detect semantic plagiarism using AI-powered sentence embeddings. This tool compares text against a reference corpus to identify paraphrased content, even when wording differs—ideal for educators, content creators, and researchers.

## Features

- ✅ Detects paraphrased content (not just copy-paste)
- ⚡ Processes texts in seconds
- 🌐 Supports 100+ languages via Sentence Transformers
- 📊 Outputs quantitative similarity scores

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/johnleshan/Plagiarism-Detector-Using-Sentence-Transformers.git
   cd Plagiarism-Detector-Using-Sentence-Transformers

2. Install dependencies:
   pip install -r requirements.txt


<p align="center">
  How it works
  <br>
</p>
graph LR
    A[Input Text Files] --> B[Generate Sentence Embeddings]
    B --> C[Calculate Cosine Similarity]
    C --> D[Output Similarity Score]