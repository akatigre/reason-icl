# vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --host 0.0.0.0 --dtype bfloat16 --max_model_len 32768 --tensor_parallel_size 4 --chat-template ./qwen.jinja
# python3 vllm_run.py --dataset_name MATH500 --split test --model_name Qwen/Qwen2.5-7B-Instruct
# vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --host 0.0.0.0 --dtype bfloat16 --max_model_len 32768 --tensor_parallel_size 4 --chat-template ./qwen.jinja
# python3 vllm_run.py --dataset_name MATH500 --split test --model_name Qwen/Qwen2.5-32B-Instruct
# vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000 --host 0.0.0.0 --dtype bfloat16 --limit-mm-per-prompt image=5,video=5 --max_model_len 128000 --tensor_parallel_size 4 --chat-template ./qwen_vl.jinja
python3 vllm_run.py --dataset_name MathVista --split testmini --model_name Qwen/Qwen2.5-VL-7B-Instruct
python3 vllm_run.py --dataset_name REMI --split val --model_name Qwen/Qwen2.5-VL-7B-Instruct
python3 vllm_run.py --dataset_name AOKVQA --split val --model_name Qwen/Qwen2.5-VL-7B-Instruct
