# lerbut

This project introduces a Retrieval Augmented Generation (RAG) pipeline designed for advanced question answering.

## Key Features

- **Language Model (LLM):** The pipeline leverages a locally hosted Large Language Model called Mistral 7b hosted locally by Ollama, operating within Docker containers for enhanced accessibility and management.

- **QA Framework:** Entire question answering processes are constructed using Langchain, ensuring a cohesive and adaptable system that aligns seamlessly with the academic context.

- **Embeddings Model:** The system capitalizes on the robust BGE Large En v1.5 embeddings model, enabling enriched context representation for precise comprehension and accurate responses.

## Retrieval Strategies

- **BM25 Keyword Search:** A meticulous keyword search strategy is employed to ensure comprehensive coverage and retrieval of pertinent information.

- **Semantic Search with Embeddings:** The pipeline incorporates a sophisticated semantic search methodology utilizing the embeddings model, enabling nuanced contextual understanding for refined retrieval.

## Infrastructure

- **Vectorstore Utilization:** ChromaDB serves as the vectorstore, streamlining storage and retrieval of contextual embeddings to expedite the search process.

## Performance Metrics

- **Inference Time:** The average inference time ranges between 60 to 90 seconds, striking a balance between processing speed and accuracy to deliver timely responses.

- **Accuracy:** While the current accuracy of the QA model is categorized as 'passable,' it reflects a robust foundation with potential for further enhancement and refinement.

## System Requirements**

**Hardware:**

* **GPU:** NVIDIA RTX 3050 Ti Mobile
* **CPU:** AMD Ryzen 7 5000 series
* **Memory:** 16GB RAM
* **Storage:** 500GB SSD (20GB minimum required)

**Software:**

* **Operating System:** Linux Debian-based
* **Python:** 3.10
* **CUDA:** (version)
* **cuDNN:** (version)
* **Docker:** (version)
* **Olllama:** (version)
* **nvidia-container-toolkit:** (version)
* **Python dependencies:** listed in `requirements.txt`

# Setup instructions
### 1. CUDA and CuDNN Installation:

#### CUDA Installation:

1. **Verify GPU Compatibility:**
   ```bash
   lspci | grep -i nvidia
   ```

2. **Download CUDA Toolkit (replace `<version>` with your desired CUDA version):**
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/debian10/x86_64/cuda-repo-debian10_<version>_amd64.deb
   ```

3. **Install CUDA Repository Package:**
   ```bash
   sudo dpkg -i cuda-repo-debian10_<version>_amd64.deb
   sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/debian10/x86_64/7fa2af80.pub
   sudo apt-get update
   ```

4. **Install CUDA Toolkit:**
   ```bash
   sudo apt-get install cuda
   ```

#### CuDNN Installation:

1. **Download CuDNN (requires NVIDIA Developer account):**
   Go to the [NVIDIA CuDNN page](https://developer.nvidia.com/cudnn), download the CuDNN version compatible with your CUDA version.

2. **Extract and Install CuDNN:**
   ```bash
   tar -xzvf cudnn-<version>-linux-x64-v<cuDNN_version>.tgz
   sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
   sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
   sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
   ```

### 2. Docker Installation:

#### Docker CE Installation:

1. **Install Required Packages:**
   ```bash
   sudo apt-get update
   sudo apt-get install apt-transport-https ca-certificates curl gnupg lsb-release
   ```

2. **Add Docker Repository Key:**
   ```bash
   curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
   ```

3. **Add Docker Repository:**
   ```bash
   echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   ```

4. **Install Docker Engine:**
   ```bash
   sudo apt-get update
   sudo apt-get install docker-ce docker-ce-cli containerd.io
   ```

5. **Start and Enable Docker Service:**
   ```bash
   sudo systemctl start docker
   sudo systemctl enable docker
   ```

### 3. Installation of Python Dependencies:

1. **Install Python and pip:**
   ```bash
   sudo apt-get update
   sudo apt-get install python3 python3-pip
   ```

2. **Install Required Python Packages (replace `<package>` with actual package names):**
   ```bash
   sudo pip3 install <package1> <package2> ...
   ```

### 4. NVIDIA Container Toolkit Installation:

1. **Add NVIDIA Container Toolkit Repository:**
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   ```

2. **Install NVIDIA Container Toolkit:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

### Starting the LLM (Large Language Model)

Once the setup is completed successfully, you can initiate the Large Language Model (LLM) using Docker with the following commands:

1. **Run the Ollama Container:**
   
   ```bash
   docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
   ```

   This command initiates the Ollama container, allocating all available GPUs (`--gpus=all`), creating a volume to persist data (`-v ollama:/root/.ollama`), and mapping the Ollama port (`-p 11434:11434`). It names the container as "ollama."

2. **Execute Mistral with Ollama:**

   ```bash
   docker exec -it ollama ollama run mistral
   ```

   This command uses `docker exec` to execute a command (`ollama run mistral`) within the running "ollama" container. It launches Mistral, the Large Language Model integrated with Ollama, allowing for subsequent actions and interactions with the model.
