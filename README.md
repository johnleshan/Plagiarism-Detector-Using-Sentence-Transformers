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


## Usage
The Plagiarism Detector features a simple graphical interface that makes detecting plagiarism intuitive. Follow these steps:

Step 1: Upload Files
Click the "UPLOAD FILES" button to select and upload the documents you want to check for plagiarism.

<p align="center">
<img src="assets/Upload files.png" alt="Upload Files" width="700">
</p>

Step 2: Check Plagiarism
After uploading your files, click "CHECK PLAGIARISM" to begin the analysis. The app will compare your documents against its reference corpus.

<p align="center">
<img src="assets/Check plagiarism.png" alt="Check Plagiarism" width="700">
</p>

Step 3: View Results
Once the analysis is complete, click "SHOW COPIED TEXTS" to see detailed results including similarity scores and specific text matches.

<p align="center">
<img src="assets/Show copied text.png" alt="Show Results" width="700">
</p>

Step 4: Export Report
Generate a comprehensive report of the plagiarism analysis by clicking "EXPORT REPORT". You can choose between PDF and CSV formats.

<p align="center">
<img src="assets/Export report.png" alt="Export Report" width="700">
</p>


## How it works
<p align="center">
<img src="diagram.svg" alt="Workflow Diagram">
</p>