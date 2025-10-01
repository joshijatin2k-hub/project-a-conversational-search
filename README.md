## 
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

mkdir -p data/source
echo "BMW internship notes. Conversational AI, intelligent search, cloud, APIs, data governance." > data/source/notes.txt

uvicorn app:app --reload --port 8000
# Open http://127.0.0.1:8000/docs```
