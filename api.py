from typing import Union

from fastapi import FastAPI, Request
from llm_stuff import create_qa_chain, process_llm_response

qa_chain = create_qa_chain()

app = FastAPI()


@app.post("/query")
async def query(request: Request):
    r = await request.json()
    q = r['query']
    res = await process_llm_response(q, qa_chain)
    print(res)
    r = {
        "query": q,
    }
    return r


@app.post("/hello")
async def hello(request: Request):
    r = await request.json()
    q = r['query']
    res = "Hello, " + q
    r = {
        "query": q,
        "response": res
    }
    return r


# Write a curl command to test the API
# curl -X POST -H "Content-Type: application/json" -d '{"query": "What is the purpose of the document?"}' http://localhost:8000/query
