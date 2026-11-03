VLLM SERVER SETUP
1) conda create --name MultiAgenticResearch python=3.11
2) conda activate MultiAgenticResearch
3) conda install -c conda-forge gxx_linux-64 && pip install "vllm>=0.4.0" openai flashinfer-python
4) vllm serve "Qwen/Qwen2.5-VL-32B-Instruct-AWQ" --seed 42 --max-model-len 72000




Setup
1) conda create --name mixingdeskadversarialattack python=3.11
2) conda activate mixingdeskadversarialattack
3) pip install -r requirements.txt
4) cp .env.example .env
5) Ensure vLLM is running and accessible at VLLM_BASE_URL

Run Attacking Lineup
python run_attacker.py --image img_0006.png

