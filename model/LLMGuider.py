import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time
import google.generativeai as genai
import os
import json
import re
import random
from datetime import datetime
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

class LLMGuider(nn.Module):
    """
    A module to guide the topic model's training using feedback from a Large Language Model (Gemini).
    This version implements an advanced list-wise ranking loss based on KL-Divergence.
    """
    def __init__(self,
                 lambda_contrastive: float,
                 gemini_model_name: str,
                 llm_contrastive_temperature: float,
                 llm_guidance_refresh_rate: int,
                 llm_top_k: int,
                 llm_history_length: int,
                 llm_max_retries: int,
                 llm_retry_delay: int,
                 num_times: int,
                 num_topic: int,
                 llm_batch_size: int,
                 log_path: str = "./llm_logs/"):
        super().__init__()
        
        self.lambda_contrastive = lambda_contrastive
        self.gemini_model_name = gemini_model_name
        self.llm_contrastive_temperature = llm_contrastive_temperature
        self.llm_guidance_refresh_rate = llm_guidance_refresh_rate
        self.llm_top_k = llm_top_k
        self.llm_history_length = llm_history_length
        self.llm_max_retries = llm_max_retries
        self.llm_retry_delay = llm_retry_delay
        self.num_times = num_times
        self.num_topic = num_topic
        self.llm_batch_size = llm_batch_size
        
        self.models = []
        self.model_cycler = None
        self.executor = None

        self.log_path = log_path
        if self.lambda_contrastive > 0 and self.log_path:
            os.makedirs(self.log_path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = os.path.join(self.log_path, f"llm_guidance_log_{timestamp}.jsonl")

        if self.lambda_contrastive > 0:
            self._initialize_gemini_models()
        
        self.guidance_cache = {}
        self.refined_top_words_cache = {}
        self.epoch_last_updated = -1

    def _initialize_gemini_models(self):
        """
            Initialize the Gemini models and the "call center" (ThreadPoolExecutor) for concurrent processing.
        """
        try:
            api_keys_str = os.environ.get("GOOGLE_API_KEYS")
            if not api_keys_str:
                print("FATAL: GOOGLE_API_KEYS not found. LLMGuider will be disabled.")
                return
            api_keys = [key.strip() for key in api_keys_str.split(',') if key.strip()]
            generation_config = {"response_mime_type": "application/json"}
            
            print(f"Found {len(api_keys)} API keys. Initializing Gemini models...")
            for key in api_keys:
                genai.configure(api_key=key)
                model = genai.GenerativeModel(self.gemini_model_name, generation_config=generation_config)
                self.models.append(model)
            
            if self.models:
                print(f"{len(self.models)} Gemini models initialized successfully.")
                self.model_cycler = itertools.cycle(self.models)
                self.executor = ThreadPoolExecutor(max_workers=len(self.models))
            else:
                print("Warning: No valid API keys were processed. LLMGuider will be disabled.")
        except Exception as e:
            print(f"An error occurred during Gemini model initialization: {e}. LLMGuider will be disabled.")
            self.models = []

    def _get_next_model(self):
        if not self.model_cycler: return None
        return next(self.model_cycler)
    
    def _create_batch_prompt(self, topics_in_batch: list) -> str:
        topics_str = ""
        for topic in topics_in_batch:
            historical_str = "\n".join(
                [f"      - Time t-{i+1}: {', '.join(words)}" for i, words in enumerate(topic['historical_words'])]
            ) if topic['historical_words'] else "      - No historical data available."
            candidates_str = '", "'.join([c.replace('"', '\\"') for c in topic['current_words']])
            topics_str += f"""
  {{
    "topic_index": {topic['id'][1]},
    "history": [
{historical_str}
    ],
    "candidates": ["{candidates_str}"]
  }},"""
        topics_str = topics_str.strip()[:-1]

        return f"""
        You are a meticulous and discerning topic model analyst. Your task is to analyze a batch of topics from the same time slice (time t). For each topic in the batch, you must refine its candidate keyword list based on its history with nuanced scoring.

        Here is the batch of topics to analyze:
        [
        {topics_str}
        ]

        *Your Step-by-Step Task for EACH topic in the batch:*
        1.  *Analyze & Summarize:* First, carefully compare the topic's "candidates" with its "history". In one sentence, summarize the core semantic concept of the topic at time t and note its primary evolution (e.g., "The topic has shifted from general finance to specifically focus on stock market volatility"). This summary is CRITICAL.
        2.  *Filter for Irrelevance:* From the "candidates" list, discard any words that are nonsensical, typos, or entirely unrelated to the core concept you identified.
        3.  *Score & Differentiate:* For each of the remaining, relevant keywords, assign a "novelty_score" from 0.0 to 1.0. **Crucially, you MUST provide a differentiated range of scores.** It is highly improbable that all words share the same level of novelty. Use the full 0.0 to 1.0 scale to reflect your nuanced analysis, based on the following precise criteria:
            - A score of **1.0** for a highly relevant word that is **completely new** and was NOT present in the Topic History. These are true "emerging words".
            - A score between **0.5 and 0.9** for a word that may have appeared in the history but has gained **significantly more importance or a new contextual meaning** now.
            - A score between **0.1 and 0.4** for a relevant word that is **stable** and has been consistently important in both the past and the present.
            - A score of **0.0** for a word that is still on the candidate list but you assess as **no longer relevant or outdated** for the topic's current direction.
        4.  *Format the Output:* Your response must be **ONLY** a single, valid JSON object and nothing else. **Do not include any introductory text, concluding remarks, or markdown formatting such as ```json.** The entire response should start with `{{` and end with `}}`.

        Example of the required JSON structure for the entire batch:
        {{
          "analyzed_topics": [
            {{
              "topic_index": 0,
              "reasoning_summary": "The topic has evolved from general machine learning concepts to focus specifically on the components of transformer architectures.",
              "ranked_candidates": [
                {{ "word": "transformer", "novelty_score": 1.0 }},
                {{ "word": "attention", "novelty_score": 0.8 }},
                {{ "word": "model", "novelty_score": 0.2 }}
              ]
            }}
          ]
        }}
        """

    def _parse_batch_output(self, text: str) -> dict | None:
        try:
            start_index = text.find('{')
            end_index = text.rfind('}')
            
            if start_index == -1 or end_index == -1 or end_index < start_index:
                print("    -> Parser Warning: Could not find valid JSON start/end braces.")
                return None

            json_str = text[start_index : end_index + 1]
            data = json.loads(json_str)
            
            if "analyzed_topics" in data and isinstance(data["analyzed_topics"], list):
                results_dict = {
                    item['topic_index']: {
                        'reasoning_summary': item.get('reasoning_summary', 'N/A'),
                        'ranked_candidates': item.get('ranked_candidates', [])
                    } for item in data['analyzed_topics'] if 'topic_index' in item
                }
                return results_dict
            else:
                print("    -> Parser Warning: 'analyzed_topics' key not found or not a list in JSON.")
                return None
        except (json.JSONDecodeError, AttributeError, KeyError) as e:
            print(f"    -> Parser Error: {e}")
            return None

    def _get_guidance_from_api_batched(self, batch_of_topics: list, current_time: int) -> dict:
        if not self.models or not batch_of_topics: return {}
        
        prompt = self._create_batch_prompt(batch_of_topics)
        result_dict = None
        
        current_model = self._get_next_model()
        if not current_model: return {}

        for attempt in range(self.llm_max_retries):
            try:
                start_time = time.time()
                response = current_model.generate_content(prompt)
                end_time = time.time()
                duration = end_time - start_time
                
                parsed_result = self._parse_batch_output(response.text)

                if self.log_path:
                    with open(self.log_file, 'a', encoding='utf-8') as f:
                        log_entry = {
                            "timestamp": datetime.now().isoformat(),
                            "time_slice": current_time,
                            "topic_indices_in_batch": [t['id'][1] for t in batch_of_topics],
                            "duration_seconds": duration,
                            "prompt_length_chars": len(prompt),
                            "prompt": prompt,
                            "raw_response": response.text,
                            "parsed_successfully": parsed_result is not None
                        }
                        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

                if parsed_result is not None: 
                    parsed_result['__meta__'] = {'duration': duration}
                    return parsed_result
                
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                print(f"    -> Batch API call for t={current_time} failed after {duration:.2f}s (attempt {attempt+1}/{self.llm_max_retries}): {e}")
                if attempt < self.llm_max_retries - 1: time.sleep(self.llm_retry_delay)
        
        return {}

    def _chunk_list(self, lst, n):
        """Splits a list into sublists of size n."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def update_guidance_cache(self, current_epoch: int, beta: torch.Tensor, idx_to_word: dict):
        if not self.executor or self.llm_guidance_refresh_rate <= 0 or current_epoch % self.llm_guidance_refresh_rate != 0 or current_epoch == self.epoch_last_updated:
            return

        print(f"\n--- Epoch {current_epoch}: Refreshing LLM Guidance Cache (Concurrent Mini-Batched) ---")
        
        for t in range(1, self.num_times):
            all_topics_for_slice = []
            for k in range(self.num_topic):
                beta_k_t = beta[t, k, :]
                _, top_indices = torch.topk(beta_k_t, self.llm_top_k)
                current_words = [idx_to_word[idx.item()] for idx in top_indices]
                historical_words = []
                for hist_t in range(max(0, t - self.llm_history_length), t):
                    beta_k_hist_t = beta[hist_t, k, :]
                    _, top_indices_hist = torch.topk(beta_k_hist_t, self.llm_top_k)
                    historical_words.append([idx_to_word[idx.item()] for idx in top_indices_hist])
                
                all_topics_for_slice.append({
                    "id": (t, k), 
                    "current_words": current_words, 
                    "historical_words": historical_words
                })
            
            mini_batches = list(self._chunk_list(all_topics_for_slice, self.llm_batch_size))
            print(f"Time slice {t}: Submitting {len(mini_batches)} mini-batches to the executor...")

            futures = [self.executor.submit(self._get_guidance_from_api_batched, batch, t) for batch in mini_batches]

            with tqdm(total=len(futures), desc=f"  Processing for t={t}") as pbar:
                for future in as_completed(futures):
                    guidance_results_dict = future.result()
                    
                    duration = guidance_results_dict.pop('__meta__', {}).get('duration', 0)
                    pbar.set_postfix_str(f"last call: {duration:.2f}s")
                    
                    for k_result, result_data in guidance_results_dict.items():
                        topic_id = (t, k_result)
                        ranked_candidates = result_data.get('ranked_candidates', [])
                        
                        self.guidance_cache[topic_id] = ranked_candidates
                        
                        try:
                            sorted_candidates = sorted(
                                ranked_candidates, 
                                key=lambda x: x.get("novelty_score", 0.0), 
                                reverse=True
                            )
                            self.refined_top_words_cache[topic_id] = [item["word"] for item in sorted_candidates[:15]]
                        except (TypeError, KeyError):
                             print(f"Warning: Could not process ranked_candidates for topic {topic_id}")
                    pbar.update(1)
        
        self.epoch_last_updated = current_epoch
        print(f"--- LLM Guidance Cache Updated ---")

    def calculate_contrastive_loss(self, topic_embeddings: torch.Tensor, word_embeddings: torch.Tensor, word_to_idx: dict) -> torch.Tensor:
        """Calculate the list-wise ranking loss."""
        if not self.guidance_cache or self.lambda_contrastive <= 0:
            return torch.tensor(0.0, device=topic_embeddings.device)

        total_loss = 0.0
        num_guided_topics = 0
        
        for (t, k), ranked_candidates in self.guidance_cache.items():
            if not ranked_candidates or len(ranked_candidates) < 2:
                continue

            topic_emb = topic_embeddings[t, k]
            
            valid_words_embs = []
            target_scores = []
            for cand in ranked_candidates:
                if cand.get("word") in word_to_idx:
                    valid_words_embs.append(word_embeddings[word_to_idx[cand["word"]]])
                    target_scores.append(cand.get("novelty_score", 0.0))

            if len(valid_words_embs) < 2:
                continue
            
            candidate_embs = torch.stack(valid_words_embs)
            target_scores = torch.tensor(target_scores, device=topic_emb.device)

            model_sims = F.cosine_similarity(topic_emb.unsqueeze(0), candidate_embs)
            model_dist = F.log_softmax(model_sims / self.llm_contrastive_temperature, dim=-1)

            target_dist = F.softmax(target_scores / self.llm_contrastive_temperature, dim=-1).detach()
            
            kl_loss = F.kl_div(model_dist, target_dist, reduction='sum')
            
            total_loss += kl_loss
            num_guided_topics += 1

        return total_loss / (num_guided_topics + 1e-8)
