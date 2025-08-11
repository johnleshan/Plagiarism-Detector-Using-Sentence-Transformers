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


## How it works
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 300">
  <style>
    .box { fill: #4A90E2; stroke: #2E5C8A; stroke-width: 2; rx: 10; }
    .text { fill: white; font-family: Arial, sans-serif; font-size: 16px; text-anchor: middle; }
    .arrow { stroke: #333; stroke-width: 2; fill: none; marker-end: url(#arrowhead); }
  </style>
  
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333" />
    </marker>
  </defs>
  
  <!-- Input Text Files -->
  <rect class="box" x="20" y="120" width="160" height="60"/>
  <text class="text" x="100" y="155">Input Text Files</text>
  
  <!-- Generate Sentence Embeddings -->
  <rect class="box" x="220" y="120" width="200" height="60"/>
  <text class="text" x="320" y="145">Generate Sentence</text>
  <text class="text" x="320" y="165">Embeddings</text>
  
  <!-- Calculate Cosine Similarity -->
  <rect class="box" x="460" y="120" width="200" height="60"/>
  <text class="text" x="560" y="145">Calculate Cosine</text>
  <text class="text" x="560" y="165">Similarity</text>
  
  <!-- Output Similarity Score -->
  <rect class="box" x="700" y="120" width="160" height="60"/>
  <text class="text" x="780" y="145">Output Similarity</text>
  <text class="text" x="780" y="165">Score</text>
  
  <!-- Arrows -->
  <path class="arrow" d="M180,150 L220,150"/>
  <path class="arrow" d="M420,150 L460,150"/>
  <path class="arrow" d="M660,150 L700,150"/>
</svg>