# Qualitative Analysis of Theory of Mind Failures in LLMs

## Overview

This project investigates **how and why** large language models fail at Theory of Mind (ToM) tasks, moving beyond simple accuracy metrics to understand the underlying reasoning breakdowns. ToM refers to the ability to attribute mental states (e.g., beliefs, desires, intentions) to others and predict their behavior accordingly.

## Approach

1. **Benchmark Evaluation**: Test 3-5 state-of-the-art LLMs (GPT, Claude, Gemini, Qwen, Llama) on ToM-Bench (Chen et al., 2024), covering 8 core ToM tasks and 31 cognitive abilities
2. **Failure Analysis**: Extract and analyze reasoning traces from failed cases using chain-of-thought prompting
3. **Hierarchical Clustering**: Apply AI-assisted summarization to classify failures into interpretable categories
4. **Targeted Improvement**: Test whether addressing identified failure patterns through prompt-tuning or adapter-based fine-tuning improves ToM performance

## Deliverables

- Comparative evaluation baseline and identified failure cases
- AI-based hierarchical summarization framework for failure classification
- Curated dataset of verified reasoning-category labels
- Validation results showing targeted interventions improve performance

