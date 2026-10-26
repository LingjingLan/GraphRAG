# Benchmark question generation and reliability

The benchmark defines four question types:
1. Fact Retrieval
2. Complex Reasoning
3. Contextual Summarize
4. Creative Generation

`benchmark/esg_qg_dataset_2015.json` is the current benchmark. The `reliability/` input contains exactly its 2,015 question IDs, each rated by two annotators. Run `python reliability/krippendorff_alpha.py` to calculate ordinal and interval Krippendorff’s α for the seven rubric dimensions.
