# Benchmark annotation application

This FastAPI application provides the annotation frontend and backend used to
score the 2,015-question benchmark on seven 1–3 rubric dimensions. It uses
`data/questions_2015.json`, matching the benchmark in the parent directory.

Run locally:

```bash
pip install -r requirements.txt
copy secrets.example.json secrets.json
uvicorn app.main:app --reload --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765/login.html`. Change all example credentials before
deploying. Runtime SQLite data, exported annotations, and `secrets.json` are
excluded from Git.
