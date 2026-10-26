# Experimental results

`evaluation/` contains compact retrieval/generation evaluation outputs and summary files for all evaluated methods:

- DualChannel
- HippoRAG / HippoRAG2
- LightRAG
- HyperGraphRAG

`raw_qa/` contains the per-question QA responses for the current ten evaluated
model configurations. These files are tracked with Git LFS because several
exceed GitHub's normal 100 MB file limit.

The 728 MB historical `_archive_pre_new_method_20260806_182752` directory is
excluded because it contains superseded experimental runs.
