import sys
import json
from client import TFIDFVectorizerEngine

def handle_rpc(line):
    try:
        req = json.loads(line)
    except Exception:
        return
    req_id = req.get("id")
    method = req.get("method")
    params = req.get("params", {})

    if method == "initialize":
        res = {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "genpark-tfidf-vectorizer-cosine-similarity-skill", "version": "1.0.0"},
            "capabilities": {"tools": {}}
        }
    elif method == "tools/list":
        res = {
            "tools": [
                {
                    "name": "compute_similarity",
                    "description": "Compute TF-IDF vectors and pairwise cosine similarity for a list of documents",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "documents": {"type": "array", "items": {"type": "string"}},
                            "doc_a_idx": {"type": "integer", "default": 0},
                            "doc_b_idx": {"type": "integer", "default": 1}
                        },
                        "required": ["documents"]
                    }
                }
            ]
        }
    elif method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})
        if tool_name == "compute_similarity":
            engine = TFIDFVectorizerEngine()
            docs = args.get("documents", [])
            idx_a = args.get("doc_a_idx", 0)
            idx_b = args.get("doc_b_idx", 1)
            vecs = engine.fit_transform(docs)
            sim = engine.cosine_similarity(vecs[idx_a], vecs[idx_b]) if len(vecs) > max(idx_a, idx_b) else 0.0
            res = {"content": [{"type": "text", "text": json.dumps({"similarity": sim, "vocab_size": len(engine.vocab)})}]}
        else:
            res = {"isError": True, "content": [{"type": "text", "text": f"Unknown tool {tool_name}"}]}
    else:
        res = {"error": {"code": -32601, "message": "Method not found"}}

    resp = {"jsonrpc": "2.0", "id": req_id, "result": res.get("result", res)}
    sys.stdout.write(json.dumps(resp) + "\n")
    sys.stdout.flush()

def main():
    for line in sys.stdin:
        if line.strip():
            handle_rpc(line.strip())

if __name__ == "__main__":
    main()
