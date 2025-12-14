from typing import List, Dict, Any
import spacy
nlp = spacy.load("en_core_web_sm")
import re


SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

def extract_claims(text: str) -> List[str]:
    """
    Improved claim extraction:
    - Split by sentence boundaries
    - Then split compound clauses (and, but, also, however)
    - Returns clean, short, atomic claims
    """
    text = text.replace("\n", " ").strip()

    sentences = SENT_SPLIT_RE.split(text)

    claims = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue

       
        parts = re.split(r"\b(?:and|but|also|however|furthermore|moreover)\b", 
                         s, flags=re.IGNORECASE)

        for p in parts:
            p = p.strip()
          
            if len(p) > 5:
                claims.append(p)

    return claims

def extract_concepts(text: str):
    """
    Extract nouns, proper nouns, verbs as general-purpose factual concepts.
    """
    doc = nlp(text)
    concepts = []
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN", "VERB"]:
            concepts.append(token.lemma_.lower())
    return set(concepts)


def semantic_score(response: str, target_texts: List[str], embed_fn) -> float:
    # embed response and targets and compute top cosine similarity
    r_vec = embed_fn([response])[0]
    t_vecs = embed_fn(target_texts)
    import numpy as np
    def cos(a,b):
        a = np.array(a); b = np.array(b)
        return float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))
    scores = [cos(r_vec, tv) for tv in t_vecs]
    return max(scores) if scores else 0.0


def evaluate_relevance_and_completeness(response: str, user_message: str, contexts: List[Dict[str,Any]], embed_fn) -> Dict[str,Any]:
    ctx_texts = [c.get('text', '') for c in contexts]

    relevance_with_user = semantic_score(response, [user_message], embed_fn)
    relevance_with_ctx = semantic_score(response, ctx_texts[:5], embed_fn)
    relevance = 0.5 * relevance_with_user + 0.5 * relevance_with_ctx
    
    covered = 0
    checks = []

    response_lower = response.lower()

    import re

    for i, ctx in enumerate(contexts[:10]):
        ctx_text = ctx.get("text", "")
        ctx_lower = ctx_text.lower()
        
        entity_match = re.search(r"(hotel\s+[a-zA-Z0-9]+)", ctx_lower)
        entity = entity_match.group(1) if entity_match else None
        if entity:
            entity_in_response = entity in response_lower
            covered_flag = entity_in_response

            sem = semantic_score(ctx_text, [response], embed_fn)
        else:
            sem = semantic_score(ctx_text, [response], embed_fn)
            covered_flag = sem > 0.70  

        checks.append({
            "index": i,
            "entity": entity,
            "semantic_score": sem,
            "covered": covered_flag
        })

        if covered_flag:
            covered += 1

    completeness = covered / max(1, len(checks))

    return {
        "relevance_score": round(relevance, 3),
        "similarity_with_user": round(relevance_with_user, 3),
        "similarity_with_contexts_top5": round(relevance_with_ctx, 3),
        "completeness_score": round(completeness, 3),
        "coverage": checks
    }


def detect_hallucinations(response: str, contexts: List[Dict[str,Any]], embed_fn):

    claims = extract_claims(response)
    ctx_texts = [c.get("text", "") for c in contexts]
    context_concepts = set()
    for ctx in ctx_texts:
        context_concepts |= extract_concepts(ctx.lower())

    unsupported = []

    for claim in claims:
        sem = semantic_score(claim, ctx_texts, embed_fn)
        semantic_support = sem > 0.80   

        claim_concepts = extract_concepts(claim.lower())

        overlap = len(claim_concepts & context_concepts)
        high_overlap = overlap >= 1       

        new_entities = claim_concepts - context_concepts
        introduces_new_concept = len(new_entities) > 0
        
        if len(new_entities) == 0:
            supported = True
        else:
            supported = semantic_support and high_overlap

        if not supported:
            unsupported.append({
                "claim": claim,
                "semantic_score": round(sem, 3),
                "concepts": list(claim_concepts),
                "overlap_with_context": list(claim_concepts & context_concepts),
                "new_concepts": list(new_entities)
            })

    halluc_rate = len(unsupported) / max(1, len(claims))

    return {
        "claims_extracted": claims,
        "unsupported_claims": unsupported,
        "hallucination_rate": round(halluc_rate, 3)
    }

    claims = extract_claims(response)
    ctx_texts = [c.get("text", "") for c in contexts]

    context_concepts = set()
    for ctx in ctx_texts:
        context_concepts |= extract_concepts(ctx.lower())

    unsupported = []

    for claim in claims:

        sem = semantic_score(claim, ctx_texts, embed_fn)
        semantic_support = sem > 0.80

        claim_concepts = extract_concepts(claim.lower())
        overlap = len(claim_concepts & context_concepts)
        high_overlap = overlap >= 2

        new_entities = claim_concepts - context_concepts
        introduces_new_concept = len(new_entities) > 0
        supported = semantic_support and (high_overlap or not introduces_new_concept)

        if not supported:
            unsupported.append({
                "claim": claim,
                "semantic_score": round(sem, 3),
                "concepts": list(claim_concepts),
                "overlap_with_context": list(claim_concepts & context_concepts),
                "new_concepts": list(new_entities)
            })

    halluc_rate = len(unsupported) / max(1, len(claims))

    return {
        "claims_extracted": claims,
        "unsupported_claims": unsupported,
        "hallucination_rate": round(halluc_rate, 3)
    }

    claims = extract_claims(response)
    ctx_texts = [c.get("text", "") for c in contexts]

    context_concepts = set()
    for ctx in ctx_texts:
        context_concepts |= extract_concepts(ctx.lower())

    unsupported = []

    for claim in claims:

        sem = semantic_score(claim, ctx_texts, embed_fn)

        semantic_support = sem > 0.80 
        claim_concepts = extract_concepts(claim.lower())
        overlap = len(claim_concepts & context_concepts)

        high_overlap = overlap >= 2  
        new_entities = claim_concepts - context_concepts
        introduces_new_concept = len(new_entities) > 0
        if len(new_entities) == 0:

            supported = True
        else:
            supported = semantic_support and high_overlap

        if not supported:
            unsupported.append({
                "claim": claim,
                "semantic_score": round(sem, 3),
                "concepts": list(claim_concepts),
                "overlap_with_context": list(claim_concepts & context_concepts),
                "new_concepts": list(new_entities)
            })

    halluc_rate = len(unsupported) / max(1, len(claims))

    return {
        "claims_extracted": claims,
        "unsupported_claims": unsupported,
        "hallucination_rate": round(halluc_rate, 3)
    }


def grade_response(relevance: float, completeness: float, halluc_rate: float) -> Dict[str, str]:


    if relevance >= 0.75 and completeness >= 0.75 and halluc_rate == 0:
        return {
            "grade": "A",
            "reason": "Excellent relevance, full completeness, and zero hallucination."
        }

    if relevance >= 0.60 and completeness >= 0.50 and halluc_rate <= 0.25:
        return {
            "grade": "B",
            "reason": "Good relevance and acceptable completeness with low hallucination."
        }

    if relevance >= 0.40 and completeness >= 0.25 and halluc_rate <= 0.40:
        return {
            "grade": "C",
            "reason": "Moderate relevance and partial completeness with low or zero hallucination."
        }

    if relevance < 0.40 or halluc_rate > 0.40:
        return {
            "grade": "D",
            "reason": "Low relevance or high hallucination reduces answer quality."
        }

    return {
        "grade": "F",
        "reason": "Severe hallucination or invalid response."
    }
