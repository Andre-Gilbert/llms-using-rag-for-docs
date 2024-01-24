# llms-using-rag-for-docs

## Abstract

## Experiments & Results

## Get Started

Please make sure you use 3.9 < Python version < 3.12. To install the packages run:

```
pip install -r requirements.txt
```

Please check out:

- [Demo](demo.ipynb) for a showcase on how to use an AI agent.
- [Evaluation](evaluation.ipynb) for a glimpse on the automated code generation evaluation.
- [Slides](docs/NLP%20Project%20Slides.pdf) to find the presentation on this topic.
- [Paper]() to find the project report.

As everything has been implemented from scratch the code includes the following:

- [ReActAgent](llms/agents) for an implementation of ReAct prompting.
- [LLM clients](llms/clients/) for the API clients.
- [Auto evaluation](llms/evaluation/) for the automated code evaluation using a config grid. When running Auto Evaluation, embeddings will be generated automatically. The results of the run can be found in the [results](results) folder and the details of the run can be found in [details](results/details/) folder.
- [RAG/CoALA](llms/rag) for an implementation of RAG/CoALA using FAISS.
