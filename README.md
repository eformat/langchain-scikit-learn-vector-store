# Parquet Vector Store

- https://python.langchain.com/docs/integrations/vectorstores/sklearn/

Setup venv

```bash
python3 -m venv venv
source venv/bin/activate
```

Install deps

```bash
pip install --upgrade --quiet scikit-learn
pip install --upgrade --quiet bson
pip install --upgrade --quiet pandas pyarrow
pip install --upgrade --quiet langchain_community langchain_text_splitters langchain_huggingface
pip install --upgrade --quiet parquet-tools
```

OR

```bash
pip install -r requirements.txt
```

Run the example

```bash
python parquet-vector-store.py
```

Inspect the vector store

```bash
parquet-tools inspect /tmp/union.parquet
parquet-tools show --head 1 /tmp/union.parquet
```
