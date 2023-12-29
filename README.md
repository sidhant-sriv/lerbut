# lerbut

Going to be my document inference thingy, summarizer and other stuff.

# How to run (local system)

### Get Ollama docker container

Make sure to have docker installed and running on your system. And nvidia-container-toolkit if you want to use GPU.
Make sure to also have CUDA toolkit installed on your system. CUDNN as well. 
`docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama`

Mistral can be replaced with any other LLM model.
`docker exec -it ollama ollama run mistral`

### Install requirements

`pip install -r requirements.txt`
