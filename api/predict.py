from http.server import BaseHTTPRequestHandler
import json
import os

DIR = os.path.dirname(__file__)
MODEL_PATHS = [
    os.path.join(DIR, "..", "output", "model.json"),
    os.path.join(DIR, "..", "public", "output", "model.json"),
    os.path.join(DIR, "model.json")
]

def load_tree():
    for p in MODEL_PATHS:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    return None

def traverse_tree(node, sample):
    curr = node
    path = []
    is_fallback = False
    fallback_reason = ""
    while not curr.get("is_leaf", False):
        attr = curr.get("attribute")
        val = str(sample.get(attr, ""))
        path.append({
            "node_id": curr.get("node_id"),
            "attribute": attr,
            "checked_value": val,
            "entropy": curr.get("entropy", 0.0),
            "samples_count": curr.get("samples_count", 0)
        })
        children = curr.get("children", {})
        if val in children:
            curr = children[val]
        else:
            is_fallback = True
            fallback_reason = f"Giá trị '{val}' của thuộc tính '{attr}' không có nhánh tương ứng."
            break
            
    if is_fallback:
        pred = curr.get("majority_class", "unacc")
    else:
        pred = curr.get("prediction", "unacc")
        path.append({
            "node_id": curr.get("node_id"),
            "is_leaf": True,
            "prediction": pred,
            "entropy": curr.get("entropy", 0.0),
            "samples_count": curr.get("samples_count", 0)
        })
    return pred, path, is_fallback, fallback_reason

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')
        try:
            sample = json.loads(body)
            model_data = load_tree()
            if not model_data:
                raise FileNotFoundError("Không tìm thấy model.json")
                
            pred, path, fb, fb_reason = traverse_tree(model_data["tree"], sample)
            resp = {
                "prediction": pred,
                "path": path,
                "is_fallback": fb,
                "fallback_reason": fb_reason
            }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(resp, ensure_ascii=False).encode('utf-8'))
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}, ensure_ascii=False).encode('utf-8'))

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok", "message": "Car ID3 Classification API on Vercel is running"}).encode('utf-8'))
