from typing import Union, Optional
import warnings
import time
import torch
import numpy as np

from llama_cpp import Llama
from . import LanguageModel, GenerateOutput

class LlamaCppModel(LanguageModel):
    def __init__(
        self,
        model_path: str,
        max_batch_size: int,
        max_new_tokens: int,
        max_seq_len: int = 32768,
        device: str = 'cuda',
        temperature: float = 1.0,
        top_k: float = 1.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        critic=False,
        **kwargs,
    ):
        """
        Initialize a llama-cpp-python model instance. The parameters here mimic what was done with ExLlama,
        but now we rely on llama-cpp's internal configuration.
        """

        # llama-cpp-python automatically handles CPU/GPU based on 'n_gpu_layers' etc.
        # Device selection is less direct; specify GPU layers at model load if desired.
        # `max_seq_len` in llama-cpp is defined by the model itself, 
        # but can be hinted with `context_size`.
        
        # self.llm = Llama(
        #     model_path=model_path,
        #     n_ctx=max_seq_len,
        #     n_gpu_layers=100 if device.startswith("cuda") else 0,  # Adjust for GPU offloading
        #     logits_all=True,  # Allow access to all logits for get_next_token_logits and get_loglikelihood
        #     verbose=False
        # )
        self.llm = Llama.from_pretrained(
            repo_id="Qwen/Qwen2-0.5B-Instruct-GGUF",
            filename="*q8_0.gguf",
            n_ctx=max_seq_len,
            logits_all=True,  # Allow access to all logits for get_next_token_logits and get_loglikelihood
            verbose=False
        )

        self.temperature = temperature
        self.top_k = int(top_k) if top_k > 1 else 40  # llama-cpp requires int top_k, set a default if top_k ≤ 1
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.device = device
        self.max_batch_size = max_batch_size
        self.max_new_tokens = max_new_tokens
        self.max_seq_length = max_seq_len
        self.critic = critic

    def _tokenize(self, text: Union[str, list[str]], add_bos: bool=True, add_eos: bool=False):
        if isinstance(text, str):
            text = [text]
        # llama-cpp's .tokenize returns a list of token IDs.
        # Note: It does not automatically add BOS or EOS tokens.
        # Typically the BOS token in LLaMA is ID=1. EOS is ID=2 for many models.
        token_ids_batch = []
        for t in text:
            token_ids = self.llm.tokenize(t, add_bos=add_bos)
            if add_eos:
                token_ids.append(self.llm.token_eos())
            token_ids_batch.append(token_ids)
        max_len = max(len(toks) for toks in token_ids_batch)
        # Pad to max length if needed:
        padded = np.full((len(token_ids_batch), max_len), self.llm.token_eos(), dtype=np.int32)
        for i, toks in enumerate(token_ids_batch):
            padded[i, :len(toks)] = toks
        return padded

    def generate(
        self,
        inputs: list[str],
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        do_sample: bool = False,
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        token_repetition_penalty=1.0,
        num_return_sequences: int = 1,
        eos_token_id: Union[None, str, int, list[Union[str, int]]] = None,
        hide_input: bool = True,
        output_log_probs: bool = False,
        strategy="greedy",
        eos_list=None,
        **kwargs,
    ) -> GenerateOutput:
        """
        Generate completions for a list of input strings using llama-cpp-python.
        """

        if max_length is None:
            max_length = self.max_new_tokens
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        # For llama-cpp, we use `self.llm` call directly.
        # Only single-inference at a time is currently well-supported in llama-cpp-python for generation.
        # If batching is required, you will need to loop over inputs or try a newer version with better batching.
        
        decoded_list = []
        for inp in inputs:
            start_time = time.time()
            prompt = inp if not hide_input else inp

            # Construct stopping criteria if any
            stop = eos_list if eos_list is not None else None
            # Convert `top_k` to int if necessary
            top_k_val = int(top_k) if top_k > 1 else 40
            
            # Call llama to generate text
            output = self.llm(
                prompt=prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k_val,
                top_p=top_p,
                repeat_penalty=token_repetition_penalty,
                stop=stop
            )

            decoded = output["choices"][0]["text"]
            if decoded is None:
                decoded = ""
            decoded_list.append(decoded.strip())
            generate_time = time.time() - start_time
            print(f"generate takes: {generate_time}s, length: {len(decoded)}, speed: {generate_time/ (len(decoded)+1e-9)}s/char")

        return GenerateOutput(decoded_list, None)

    @torch.no_grad()
    def get_next_token_logits(
        self, 
        prompt: Union[str, list[str]],
        candidates: Union[list[str], list[list[str]]]
    ) -> list[np.ndarray]:
        """
        Return the logits for the candidate tokens given a prompt.
        """
        if isinstance(prompt, str):
            prompt = [prompt]

        # Ensure candidates is list of lists
        if isinstance(candidates[0], str):
            candidates = [candidates] * len(prompt)

        # Tokenize prompt
        prompt_tokens = self._tokenize(prompt, add_bos=True, add_eos=False)
        # Evaluate model on prompt tokens
        # llama-cpp: we can use .eval() for logits on each token
        # We want the logits at the last position, so we run eval on the entire prompt.
        # Note: llama-cpp-python returns logits after each eval call. If prompt too long, might need chunking.

        # We'll handle one prompt at a time since llama-cpp doesn't batch easily:
        # (For batch processing, run eval per prompt in a loop.)
        results = []
        for i, p_toks in enumerate(prompt_tokens):
            # Trim padding
            valid_len = (p_toks != self.llm.token_eos()).sum()
            p_toks = p_toks[:valid_len].tolist()
            # eval on these tokens
            # For llama-cpp: 
            #   If we call self.llm(prompt="...") directly, we get generation.
            #   Instead, use self.llm.eval(tokens, ...).
            # We must reset context or use eval directly.
            
            # Evaluate tokens (excluding the last token to get logits for next token)
            # logits_all=True was enabled at init, so response contains logits at each token position.
            eval_res = self.llm.eval(
                tokens=p_toks,
                stop=None,
                # The number of tokens we feed is the full prompt. We only want the last token's logits,
                # but llama-cpp returns logits after each token. We'll just index into the last one.
            )
            
            # The final logits will be in eval_res["logits"][len(p_toks)-1]
            # Check candidate tokens
            # Tokenize candidates
            cand_list = candidates[i]
            cand_ids = [self.llm.tokenize(c, add_bos=False) for c in cand_list]
            # If any candidate is more than one token, warn
            for c, cids in zip(cand_list, cand_ids):
                if len(cids) != 1:
                    warnings.warn(f"Candidate '{c}' corresponds to {len(cids)} tokens, expected 1.")
            
            last_logits = eval_res["logits"][-1]  # the last set of logits after processing prompt_tokens
            # Extract logits for candidate tokens
            cand_logits = []
            for cids in cand_ids:
                if len(cids) == 1:
                    cand_logits.append(last_logits[cids[0]])
                else:
                    # If multiple tokens, we only take the first token's logit?
                    # Or sum them? The original code expects single token candidates.
                    cand_logits.append(last_logits[cids[0]])
            
            results.append(np.array(cand_logits))
        
        return results

    @torch.no_grad()
    def get_loglikelihood(
        self, 
        prefix: str, 
        contents: list[str],
        **kwargs
    ) -> np.ndarray:
        """
        Compute the log-likelihood of `contents` given `prefix`.
        This involves evaluating the probability of each token in contents after prefix.
        """

        # Combine prefix and content and compute log likelihood
        full_texts = [prefix + c for c in contents]
        tokens_batch = self._tokenize(full_texts, add_bos=True, add_eos=False)
        prefix_tokens = self._tokenize([prefix], add_bos=True, add_eos=False)[0]

        prefix_len = (prefix_tokens != self.llm.token_eos()).sum()

        # We'll compute log_probs for each token beyond prefix
        # Since llama-cpp doesn't batch easily, handle one by one:
        loglikelihoods = []
        for i, seq in enumerate(tokens_batch):
            valid_len = (seq != self.llm.token_eos()).sum()
            seq_tokens = seq[:valid_len].tolist()

            eval_res = self.llm.eval(tokens=seq_tokens, stop=None)

            # eval_res["logits"] shape: [T, vocab_size]
            # We get probabilities token-by-token.
            # The probability of token i is based on logits at i-1.
            # So for token at position i (0-indexed), we look at logits[i-1].
            # Start computing after prefix_len.
            ll = 0.0
            for pos in range(prefix_len, len(seq_tokens)):
                token_id = seq_tokens[pos]
                logits = eval_res["logits"][pos-1]  # logits predicting token at pos
                # softmax to get probability
                probs = np.exp(logits - np.logaddexp.reduce(logits))
                ll += np.log(probs[token_id] + 1e-9)
            
            loglikelihoods.append(ll)

        return np.array(loglikelihoods)

    def reward(self, 
               prefix: str, 
               full_contents: list[str],
               reward_model: str,
               **kwargs) -> np.ndarray:
        """
        This method in the original code uses a second model (draft_model) and complex calculations.
        Here, we only provide a placeholder.

        Implementing this fully would require:
        - Loading another Llama model for draft/critic if needed.
        - Computing KL-divergence or JS-divergence from logits.
        - Computing loglikelihood differences.

        For now, we just return a dummy reward.
        """
        # Placeholder: Just return a dictionary with some dummy values.
        return {
            "control": 1,
            "verifier": 0,
            "sc_mcts": 0,
            "loglikelihood": 0,
            "self_eval": 0,
            "cd_logprobs_diff": 0,
            "kl_div_mean": 0,
            "kl_div_batchmean": 0,
            "js_div_clamp_batchmean": 0,
            "js_div_clamp_mean": 0,
            "js_div_batchmean": 0,
            "js_div_mean": 0,
            "intuition": 0
        }
