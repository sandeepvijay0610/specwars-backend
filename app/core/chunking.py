import hashlib
from typing import List, Dict, Any

class ChunkResult:
    def __init__(self, category: str, content: str, content_hash: str, source_url: str = None):
        self.category = category
        self.content = content
        self.content_hash = content_hash
        self.source_url = source_url

def generate_content_hash(text: str) -> str:
    """Generate a deterministic SHA-256 hash for chunk idempotency."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def generate_chunks_from_specs(phone_name: str, specs_json: Dict[str, Any]) -> List[ChunkResult]:
    """
    Deterministically creates chunks from a phone's structured specs.
    Does not use naive arbitrary character splitting.
    """
    chunks = []
    
    # Categories to map from specs
    mapping = {
        "Performance": ["processor", "ram", "storage"],
        "Camera": ["camera"],
        "Display": ["display"],
        "Battery": ["battery"],
        "Build": ["build"]
    }
    
    for category_name, fields in mapping.items():
        content_lines = [f"Phone: {phone_name}", f"Category: {category_name}"]
        has_data = False
        for field in fields:
            val = specs_json.get(field)
            if val:
                content_lines.append(f"{field.capitalize()}: {val}")
                has_data = True
                
        if has_data:
            content_text = "\n".join(content_lines)
            content_hash = generate_content_hash(content_text)
            chunks.append(ChunkResult(category_name, content_text, content_hash))
            
    # Process review sentiments separately
    sentiments = specs_json.get("review_sentiments", [])
    for sentiment in sentiments:
        content_text = f"Phone: {phone_name}\nCategory: Review\nSentiment: {sentiment}"
        content_hash = generate_content_hash(content_text)
        chunks.append(ChunkResult("Review", content_text, content_hash))

    return chunks

def generate_chunks_from_tavily_results(phone_name: str, results: List[Dict[str, str]]) -> List[ChunkResult]:
    """
    Deterministically creates 'Source' chunks directly from Tavily snippets,
    preserving the URL for each snippet so they can be cited in RAG.
    """
    chunks = []
    for res in results:
        url = res.get("url", "")
        content = res.get("content", "").strip()
        if not content:
            continue
            
        content_text = f"Phone: {phone_name}\nCategory: Source\nSource URL: {url}\nContent: {content}"
        content_hash = generate_content_hash(content_text)
        chunks.append(ChunkResult("Source", content_text, content_hash, source_url=url))
        
    return chunks
