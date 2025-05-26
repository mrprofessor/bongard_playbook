# Generate Captions
python scripts/captions.py --vlm instructBLIP
python scripts/captions.py --vlm llava

# BLIP
python scripts/vlm_llm_single.py --vlm blip --llm llama
python scripts/simple_eval.py --vlm blip --llm llama

python scripts/vlm_llm_single.py --vlm blip --llm qwen3
python scripts/simple_eval.py --vlm blip --llm qwen3

# llava
python scripts/vlm_llm_single.py --vlm llava --llm llama
python scripts/simple_eval.py --vlm llava --llm llama

python scripts/vlm_llm_single.py --vlm llava --llm qwen3
python scripts/simple_eval.py --vlm llava --llm qwen3

python scripts/vlm_llm_single.py --vlm llava --llm deepseek
python scripts/simple_eval.py --vlm llava --llm deepseek


# instructBLIP (garbage captions)
python scripts/vlm_llm_single.py --vlm instructBLIP --llm llama
python scripts/simple_eval.py --vlm instructBLIP --llm llama

python scripts/vlm_llm_single.py --vlm instructBLIP --llm deepseek
python scripts/simple_eval.py --vlm instructBLIP --llm deepseek

python scripts/vlm_llm_single.py --vlm instructBLIP --llm gpt41
python scripts/simple_eval.py --vlm instructBLIP --llm gpt41
