import json
import time
from typing import List, Dict, Any
from .embeddings import Embedder
from .vector_store import FaissStore
from .metrics import evaluate_relevance_and_completeness, detect_hallucinations
from .metrics import grade_response  


class Evaluator:
    def __init__(self, embed_model: str = 'all-MiniLM-L6-v2'):
        self.embedder = Embedder(embed_model)
        self.dim = 384

    def load_data(self, conv_path: str, ctx_path: str):
        self.conversation = json.load(open(conv_path, 'r', encoding='utf-8'))
        self.contexts = json.load(open(ctx_path, 'r', encoding='utf-8'))

    def build_index(self):
        texts = [c['text'] for c in self.contexts]
        vecs = self.embedder.encode(texts)
        self.store = FaissStore(self.dim)
        self.store.add(vecs, self.contexts)

    def evaluate(self) -> Dict[str,Any]:
        messages = self.conversation.get('messages', [])
        user_msgs = [m for m in messages if m.get('role')=='user']
        assistant_msgs = [m for m in messages if m.get('role')=='assistant']
        if not user_msgs:
            raise ValueError('No user message in conversation')
        user_msg = user_msgs[-1]['content']
        if not assistant_msgs:
            raise ValueError('No assistant response found in conversation')
        assistant_resp = assistant_msgs[-1]['content']

        t0 = time.time()
        rel = evaluate_relevance_and_completeness(
            assistant_resp, user_msg, self.contexts, self.embedder.encode
        )
        hall = detect_hallucinations(
            assistant_resp, self.contexts, self.embedder.encode
        )
        t1 = time.time()

       
        grade = grade_response(
            rel["relevance_score"],
            rel["completeness_score"],
            hall["hallucination_rate"]
        )

        input_tok = max(1, len((user_msg + ' '.join([c.get('text','') for c in self.contexts[:5]])).split())//0.75)
        output_tok = max(1, len(assistant_resp.split())//0.75)
        cost_est = round((input_tok/1000.0)*0.0015 + (output_tok/1000.0)*0.002, 6)

        report = {
            'relevance_and_completeness': rel,
            'hallucination_analysis': hall,
            'grade': grade,  
            'evaluation_wall_time_seconds': round(t1-t0,4),
            'token_estimate_input': input_tok,
            'token_estimate_output': output_tok,
            'estimated_cost_usd': cost_est,
        }
        return report
