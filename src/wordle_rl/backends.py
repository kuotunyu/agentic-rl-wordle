"""生成 backend 抽象：transformers（本機/通用）與 vLLM（Colab）。

重依賴全部延遲 import——本機 CPU 測試環境沒裝 torch/vllm 也能 import 本模組。
Qwen2.5 陷阱：內建 generation_config 預設帶採樣（temperature 0.7/top_p 0.8/top_k 20），
greedy 評測必須顯式關閉，否則不可重現。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GenConfig:
    max_new_tokens: int = 160
    do_sample: bool = False          # False = greedy（評測/baseline）
    temperature: float = 1.0         # 只在 do_sample=True 時生效
    top_p: float = 1.0
    stop: tuple[str, ...] = ()       # 例如 ("</guess>",)；命中時停止並保留 stop 字串


class GenerationBackend(Protocol):
    def generate(self, chats: list[list[dict]], cfg: GenConfig) -> list[str]:
        """chats = 多組對話（build_messages 輸出），回傳每組的新生成文字。"""
        ...


class TransformersBackend:
    """HF transformers 批次生成。左 padding（decoder-only 必須）；fp16/bf16 自動。"""

    def __init__(
        self,
        model_name: str,
        adapter: str | None = None,
        device: str | None = None,
        batch_size: int = 16,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Turing（RTX 20xx）無 bf16 → CUDA 用 fp16；CPU 用 fp32
        if self.device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.to(self.device).eval()
        self.batch_size = batch_size

    def generate(self, chats: list[list[dict]], cfg: GenConfig) -> list[str]:
        import torch

        outs: list[str] = []
        for i in range(0, len(chats), self.batch_size):
            batch = chats[i : i + self.batch_size]
            prompts = [
                self.tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                for c in batch
            ]
            enc = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
            kwargs: dict = dict(max_new_tokens=cfg.max_new_tokens)
            if cfg.do_sample:
                kwargs.update(do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p)
            else:
                # 壓掉 Qwen generation_config 的採樣預設
                kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
            with torch.inference_mode():
                out = self.model.generate(
                    **enc, **kwargs, pad_token_id=self.tokenizer.pad_token_id
                )
            new_tokens = out[:, enc["input_ids"].shape[1] :]
            texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            outs.extend(self._apply_stop(t, cfg.stop) for t in texts)
        return outs

    @staticmethod
    def _apply_stop(text: str, stops: tuple[str, ...]) -> str:
        # transformers 無原生 include-stop 截斷：手動在第一個 stop 字串後截斷（保留 stop）
        cut = len(text)
        for s in stops:
            pos = text.find(s)
            if pos != -1:
                cut = min(cut, pos + len(s))
        return text[:cut]


class VLLMBackend:
    """vLLM 批次生成（Colab / Linux）。adapter 給路徑時以 LoRA 掛載。"""

    def __init__(
        self,
        model_name: str,
        adapter: str | None = None,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 2048,
    ):
        from vllm import LLM

        self.adapter = adapter
        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_lora=adapter is not None,
            max_lora_rank=32,
        )

    def generate(self, chats: list[list[dict]], cfg: GenConfig) -> list[str]:
        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature if cfg.do_sample else 0.0,
            top_p=cfg.top_p if cfg.do_sample else 1.0,
            stop=list(cfg.stop) or None,
            include_stop_str_in_output=True,
        )
        lora_request = None
        if self.adapter:
            from vllm.lora.request import LoRARequest

            lora_request = LoRARequest("adapter", 1, self.adapter)
        outputs = self.llm.chat(
            chats, params, lora_request=lora_request, use_tqdm=False
        )
        return [o.outputs[0].text for o in outputs]
