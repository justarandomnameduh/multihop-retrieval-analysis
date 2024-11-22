1. To run all the retrieval, execute:
    ``bash run_stage_1.sh``
    or you can run each file separately:
        e.g. ``python rankerA.py``

2. To evaluate a stage one's output (e.g. output/rankerA.json):
    ``python MyRetEval.py output/rankerA.json``

3. To run all the RAG, execute:
    ``bash run_rag.sh``
    or you can run eacch file separately:
        e.g. ``python RAGA.py``

4. (Optional) To re-produce all the figures and tables from the report, run all the cells in:
    - MyRetEval.ipynb
    - MyRAGEval.ipynb
    - data_visualize.ipynb