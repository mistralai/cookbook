# Indexify-Mistral Cookbooks

Indexify is an open-source engine for building fast data pipelines for unstructured data(video, audio, images and documents) using re-usable extractors for embedding, transformation and feature extraction. Indexify keeps vectordbs, structured databases(postgres) updated automatically when pipelines produce embedding or structured data.

Applications can query indexes and databases using semantic search and SQL queries.

Project - [https://github.com/tensorlakeai/indexify](https://github.com/tensorlakeai/indexify)

This folder contains cookbooks demonstrating how to leverage Indexify and Mistral's large language models for building production ready pipelines for document understanding.

## Contents

1. [PDF Entity Extraction Cookbook](pdf-entity-extraction)
2. [PDF Summarization Cookbook](pdf-summarization)

## Overview

These cookbooks showcase the integration of Indexify's structured data extraction capabilities with Mistral's advanced language models.

### PDF Entity Extraction Cookbook

Learn how to build a robust entity [extraction pipeline for PDF](pdf-entity-extraction/README.md) documents. This cookbook covers:

- Setting up Indexify and required extractors
- Creating an extraction graph for entity recognition
- Implementing the extraction pipeline
- Customizing the entity extraction process

### PDF Summarization Cookbook

Explore how to create an efficient [PDF summarization pipeline](pdf-summarization/README.md). This cookbook includes:

- Installation and setup of necessary components
- Defining an extraction graph for document summarization
- Building and running the summarization pipeline
- Tips for customizing and enhancing the summarization process

## Prerequisites

Before using these cookbooks, ensure you have:

- Create a virtual env with Python 3.9 or later
  ```shell
  python3.9 -m venv ve
  source ve/bin/activate
  ```
- pip (Python package manager)
- A Mistral API key
- Basic familiarity with Python and command-line interfaces

## Getting Started

1. Install Indexify and the required extractors as detailed in each cookbook.
2. Review the cookbooks to understand the workflow and components.
3. Follow the step-by-step instructions to implement the pipelines.
4. Experiment with customizations to tailor the solutions to your specific needs.
