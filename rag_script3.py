import os
import json
import time
import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Any
import logging
from collections import defaultdict
import hashlib
from functools import lru_cache
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# DEFAULT CONFIGURATION
DEFAULT_CONFIG = {
    "model_name": "all-MiniLM-L6-v2",
    "embeddings_file": "/home/ubuntu/scraper_deep/deep_embeddings.json",
    "index_cache_file": "/home/ubuntu/scraper_deep/faiss_index.pkl",
    "gemma_url": "http://localhost:11434/api/generate",
    "gemma_model": "gemma3:27b",
    "top_k": 3,
    "min_similarity": 0.25,
    "max_context_length": 2000,
    "cache_size": 50,
    "llm_options": {
        "temperature": 0.1,
        "top_p": 0.7,
        "top_k": 10,
        "num_predict": 300,
        "num_ctx": 2048,
        "repeat_penalty": 1.1,
        "mirostat": 2,
        "mirostat_eta": 0.3,
        "mirostat_tau": 5.0
    },
    "llm_timeout": 800
}

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

class SpeedCache:
    """Minimal cache with no frills."""
    
    def __init__(self, max_size: int = 50):
        self.cache = {}
        self.max_size = max_size
        self.keys = []
    
    def get(self, key: str) -> Optional[str]:
        return self.cache.get(key)
    
    def set(self, key: str, value: str):
        if len(self.cache) >= self.max_size:
            old_key = self.keys.pop(0)
            self.cache.pop(old_key, None)
        
        self.cache[key] = value
        self.keys.append(key)
    
    def clear(self):
        """Clear all cached items."""
        self.cache.clear()
        self.keys.clear()
    
    @staticmethod
    def hash_query(query: str) -> str:
        return hashlib.md5(query.lower().encode()).hexdigest()[:12]

class ConfigurableRAG:
    """Configurable RAG system optimized for speed and testing."""
    
    def __init__(self, config: Dict = None):
        self.config = config or DEFAULT_CONFIG.copy()
        self.model = None
        self.index = None
        self.knowledge_base = []
        self.cache = SpeedCache(self.config["cache_size"])
        self.ready = False
        self.verbose = True
        self.stats = {
            'queries': 0, 
            'cache_hits': 0, 
            'avg_time': 0, 
            'timeouts': 0,
            'errors': 0,
            'total_time': 0
        }
    
    def update_config(self, new_config: Dict):
        """Update configuration dynamically."""
        self.config.update(new_config)
        # Update cache size if changed
        if "cache_size" in new_config:
            self.cache = SpeedCache(new_config["cache_size"])
    
    def set_verbose(self, verbose: bool):
        """Control verbosity for testing."""
        self.verbose = verbose
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.stats = {
            'queries': 0, 
            'cache_hits': 0, 
            'avg_time': 0, 
            'timeouts': 0,
            'errors': 0,
            'total_time': 0
        }
    
    def get_stats(self) -> Dict:
        """Get current performance statistics."""
        return self.stats.copy()
    
    def clear_cache(self):
        """Clear the query cache."""
        self.cache.clear()
    
    def setup(self) -> bool:
        """Setup the RAG system."""
        if self.verbose:
            print("🚀 Starting Configurable RAG...")
        
        # Quick Ollama check
        try:
            response = requests.get(f"{self.config['gemma_url'].replace('/api/generate', '/api/tags')}", timeout=3)
            if self.verbose:
                print("✅ Ollama ready")
        except Exception as e:
            if self.verbose:
                print(f"❌ Ollama not available: {e}")
            return False
        
        # Load cached index first
        if self._load_cache():
            if self.verbose:
                print("✅ Loaded from cache - READY!")
            self.ready = True
            return True
        
        # Build from scratch
        if self.verbose:
            print("🔨 Building index (one-time setup)...")
        if not self._load_data():
            return False
        if not self._build_index():
            return False
        self._save_cache()
        
        self.ready = True
        if self.verbose:
            print(f"✅ RAG READY! ({len(self.knowledge_base)} chunks)")
        return True
    
    def _load_cache(self) -> bool:
        """Load everything from cache."""
        try:
            cache_file = self.config["index_cache_file"]
            if not os.path.exists(cache_file):
                return False
                
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            self.knowledge_base = data['kb']
            self.index = data['idx']
            
            # Load model separately
            self.model = SentenceTransformer(self.config["model_name"])
            
            return True
        except Exception as e:
            if self.verbose:
                print(f"Cache load failed: {e}")
            return False
    
    def _save_cache(self):
        """Save to cache."""
        try:
            data = {'kb': self.knowledge_base, 'idx': self.index}
            with open(self.config["index_cache_file"], 'wb') as f:
                pickle.dump(data, f)
            if self.verbose:
                print("💾 Cached for next time")
        except Exception as e:
            if self.verbose:
                print(f"Cache save failed: {e}")
    
    def _load_data(self) -> bool:
        """Load data with filtering."""
        try:
            with open(self.config["embeddings_file"], 'r') as f:
                data = json.load(f)
            
            self.knowledge_base = []
            
            # Process data
            items = data.items() if isinstance(data, dict) else enumerate(data)
            
            for doc_id, item in items:
                text = item.get('text', '').strip()
                embedding = item.get('embedding')
                
                # Filtering for quality
                if (text and embedding and 
                    len(text) > 100 and len(text) < 3000 and
                    any(keyword in text.lower() for keyword in ['real estate', 'property', 'investment', 'market'])):
                    
                    self.knowledge_base.append({
                        'id': str(doc_id),
                        'text': text[:self.config["max_context_length"]],
                        'embedding': np.array(embedding, dtype=np.float32),
                        'title': item.get('page_title', 'Document')[:100],
                        'url': item.get('url', '')
                    })
            
            if self.verbose:
                print(f"📚 Loaded {len(self.knowledge_base)} relevant chunks")
            return len(self.knowledge_base) > 0
            
        except Exception as e:
            if self.verbose:
                print(f"Data loading failed: {e}")
            return False
    
    def _build_index(self) -> bool:
        """Build FAISS index."""
        try:
            # Load model
            self.model = SentenceTransformer(self.config["model_name"])
            
            # Build index
            embeddings = np.vstack([chunk['embedding'] for chunk in self.knowledge_base])
            faiss.normalize_L2(embeddings)
            
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings)
            
            return True
        except Exception as e:
            if self.verbose:
                print(f"Index build failed: {e}")
            return False
    
    def _retrieve(self, query: str) -> List[Dict]:
        """Retrieve relevant chunks."""
        try:
            query_emb = self.model.encode([query], show_progress_bar=False).astype('float32')
            faiss.normalize_L2(query_emb)
            
            scores, indices = self.index.search(query_emb, self.config["top_k"])
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and score > self.config["min_similarity"]:
                    chunk = self.knowledge_base[idx].copy()
                    chunk['similarity'] = float(score)
                    results.append(chunk)
            
            return results
            
        except Exception as e:
            if self.verbose:
                print(f"Retrieval error: {e}")
            return []
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with current configuration."""
        try:
            payload = {
                "model": self.config["gemma_model"],
                "prompt": prompt,
                "stream": False,
                "options": self.config["llm_options"].copy()
            }
            
            response = requests.post(
                self.config["gemma_url"], 
                json=payload, 
                timeout=self.config["llm_timeout"]
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                return answer if answer else "Unable to generate response."
            else:
                return f"LLM error: {response.status_code}"
                
        except requests.Timeout:
            self.stats['timeouts'] += 1
            return "Response timeout - try a simpler question."
        except Exception as e:
            self.stats['errors'] += 1
            return f"Technical error: {str(e)[:100]}"
    
    def ask(self, query: str, use_cache: bool = True) -> Dict[str, Any]:
        """Process query and return structured result."""
        if not self.ready:
            return {
                "answer": "System not ready",
                "sources": [],
                "time": 0,
                "error": "System not initialized"
            }
        
        start_time = time.time()
        self.stats['queries'] += 1
        
        # Check cache
        cache_key = SpeedCache.hash_query(query)
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached:
                self.stats['cache_hits'] += 1
                return {
                    "answer": cached,
                    "sources": [],
                    "time": time.time() - start_time,
                    "error": None,
                    "cached": True
                }
        
        # Retrieve relevant chunks
        chunks = self._retrieve(query)
        
        if not chunks:
            answer = "I don't have specific information about that. Could you rephrase or ask about real estate investment topics?"
            sources = []
        else:
            # Build context
            context = ""
            for i, chunk in enumerate(chunks):
                context += f"[{i+1}] {chunk['text'][:800]}...\n\n"
            
            # Create prompt
            prompt = f"""You are an expert real estate acquisition specialist with deep knowledge of property investment, market analysis, financing strategies, and deal structuring. 

When answering questions, provide comprehensive responses that include:

1. **Direct answer** - Start with a clear response to the specific question
2. **Strategic context** - Explain the broader investment implications and market considerations
3. **Practical examples** - Include real-world scenarios, calculations, or case studies when relevant
4. **Risk assessment** - Address potential challenges, risks, or red flags
5. **Action steps** - Provide specific, actionable recommendations

Draw from ALL relevant sources provided and synthesize the information to give complete, professional advice.

{context}

Question: {query}

Provide a comprehensive, practical answer:"""
            
            # Get LLM response
            answer = self._call_llm(prompt)
            sources = [{"title": chunk["title"], "similarity": chunk["similarity"]} for chunk in chunks]
        
        # Cache successful responses
        if use_cache and len(answer) > 20 and "error" not in answer.lower():
            self.cache.set(cache_key, answer)
        
        elapsed = time.time() - start_time
        self.stats['avg_time'] = (self.stats['avg_time'] * (self.stats['queries'] - 1) + elapsed) / self.stats['queries']
        self.stats['total_time'] += elapsed
        
        return {
            "answer": answer,
            "sources": sources,
            "time": elapsed,
            "error": None if "error" not in answer.lower() and "timeout" not in answer.lower() else answer,
            "cached": False
        }
    
    def interactive_mode(self):
        """Run interactive question-answering mode."""
        if not self.ready:
            print("❌ System not ready")
            return
        
        print(f"\n{'='*60}")
        print("⚡ CONFIGURABLE RAG SYSTEM")
        print("="*60)
        print("🔥 Current Configuration:")
        print(f"  • Temperature: {self.config['llm_options']['temperature']}")
        print(f"  • Top-P: {self.config['llm_options']['top_p']}")
        print(f"  • Top-K (LLM): {self.config['llm_options']['top_k']}")
        print(f"  • Top-K (Retrieval): {self.config['top_k']}")
        print(f"  • Min Similarity: {self.config['min_similarity']}")
        print(f"  • Max Tokens: {self.config['llm_options']['num_predict']}")
        print(f"  • Mirostat Eta: {self.config['llm_options']['mirostat_eta']}")
        print(f"  • Mirostat Tau: {self.config['llm_options']['mirostat_tau']}")
        print("\n💬 Commands: 'stats', 'clear', 'config', 'quit'")
        print("="*60)
        
        while True:
            try:
                query = input("\n🔥 Ask: ").strip()
                
                if not query:
                    continue
                elif query.lower() in ['quit', 'exit', 'q']:
                    break
                elif query.lower() == 'stats':
                    s = self.stats
                    cache_rate = (s['cache_hits'] / max(s['queries'], 1)) * 100
                    print(f"\n📊 Stats:")
                    print(f"  Queries: {s['queries']}")
                    print(f"  Avg time: {s['avg_time']:.2f}s")
                    print(f"  Cache hits: {cache_rate:.0f}%")
                    print(f"  Timeouts: {s['timeouts']}")
                    print(f"  Errors: {s['errors']}")
                    continue
                elif query.lower() == 'clear':
                    self.clear_cache()
                    print("🧹 Cache cleared")
                    continue
                elif query.lower() == 'config':
                    print(f"\n⚙️ Current Config:")
                    for key, value in self.config['llm_options'].items():
                        print(f"  {key}: {value}")
                    print(f"  top_k (retrieval): {self.config['top_k']}")
                    print(f"  min_similarity: {self.config['min_similarity']}")
                    continue
                
                # Process query
                result = self.ask(query)
                
                print(f"\n📝 ANSWER\n{'-'*50}")
                print(result["answer"])
                
                if result["sources"]:
                    print(f"\n📚 SOURCES ({len(result['sources'])})")
                    for i, source in enumerate(result["sources"], 1):
                        print(f"{i}. {source['title']} (score: {source['similarity']:.2f})")
                
                cached_info = " (cached)" if result.get("cached") else ""
                print(f"{'-'*50}")
                print(f"Time: {result['time']:.2f}s{cached_info}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("👋 Goodbye!")

# Factory function - ensure this is at module level and accessible
def create_rag_instance(config: Dict = None) -> ConfigurableRAG:
    """Factory function to create a RAG instance."""
    try:
        return ConfigurableRAG(config)
    except Exception as e:
        print(f"Error creating RAG instance: {e}")
        raise

# Debug function to test imports
def test_import():
    """Test function to verify the module can be imported correctly."""
    print("✅ Module imported successfully!")
    print(f"✅ create_rag_instance function is available: {callable(create_rag_instance)}")
    print(f"✅ ConfigurableRAG class is available: {ConfigurableRAG is not None}")
    return True

def main():
    """Main function for standalone usage."""
    try:
        rag = create_rag_instance()
        
        if not rag.setup():
            print("❌ Setup failed")
            return
        
        rag.interactive_mode()
    except Exception as e:
        print(f"❌ Error in main: {e}")
        import traceback
        traceback.print_exc()

# This ensures the functions are available when imported
__all__ = ['ConfigurableRAG', 'SpeedCache', 'create_rag_instance', 'DEFAULT_CONFIG', 'test_import']

if __name__ == "__main__":
    main()