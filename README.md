# Qualitative Analysis of Theory of Mind Failures in LLMs

**ToM refers to the ability to attribute mental states (e.g., beliefs, desires, intentions) to others and predict their behavior accordingly.**

## Overview

This project investigates 
1. Are LLMs truly incapable of ToM tasks, or are current benchmarks underestimating their capabilities?
2. When LLMs fail on ToM tasks, what systematic failure modes emerge?

## Approach

1. **Benchmark Evaluation**: Test one small model Qwen2.5-7B-Instruct and 5 state-of-the-art LLMs (GPT, Claude, Gemini, Qwen, Llama) on ToM-Bench (Chen et al., 2024), covering 8 core ToM tasks and 31 cognitive abilities
2. **Failure Analysis**: Analyze and summarize failure reasons of failed cases via chain-of-thought prompting
3. **Hierarchical Clustering**: Apply AI-assisted hierarchical summarization to classify failures into interpretable categories
4. **Targeted Improvement**: Test whether identifying failure patterns and synthesizing training data accordingly for model fine-tuning improves its ToM performance

## Deliverables

- Comparative evaluation baseline and identified failure cases
- AI-based hierarchical summarization framework for failure classification
- Curated dataset of verified reason-category labels
- Validation results showing targeted interventions improve performance

### Check experiment_setup.txt for step-by-step tutorial

