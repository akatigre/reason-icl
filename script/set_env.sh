uv venv --python 3.12  && source .venv/bin/activate && uv pip install --upgrade pip
uv pip install setuptools && uv pip install flash-attn --no-build-isolation
uv pip install accelerate==1.4.0 
uv pip install bitsandbytes >=0.43.0
uv pip install datasets >=3.2.0
uv pip install deepspeed==0.16.7
uv pip install distilabel[vllm,ray,openai] >= 1.5.2
uv pip install einops >=0.8.0
uv pip install huggingface-hub[cli,hf_xet] >= 0.30.2,<1.0
uv pip install lighteval @ git+https://github.com/huggingface/lighteval.git@bb14995c4eccab5cabd450b1e509c3c898a16921
uv pip install python-dotenv
uv pip install torch==2.6.0
uv pip install transformers==4.51.3
uv pip install wandb >= 0.19.1