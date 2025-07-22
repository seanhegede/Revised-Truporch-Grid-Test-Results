# Revised Truporch Grid Test Results 
Attached are the results of a configuration grid test that I had to run on OpenWebUI's gemma3:27b model, testing the LLM at different model parameters to optimize answer quality metrics. 
The file, 'notion_scraper_final.py' shows the Python script I used to scrape all the information off of the company's Notion wiki page, for use as the knowledge base under which my gemma3:27b model could retrieve context. 
rag_script3.py is the RAG pipeline I used to execute the model for querying, and Truporch_grid_results.ipynb is the Jupyter notebook I imported the RAG file into as a package. A spreadsheet of the results can be found 
in grid_results_spreadsheet.csv. 

## Components
To hook up to the gemma3:27 API, I used an r5d. xlarge EC2 instance on AWS cloud, allowing me to pull large LLM  models remotely. Ubuntu is my AMI, and I configured the security groups to allow perms on ports 10006
(for SSH) and 8889 (Jupyter). The key and IP address is all given to you when you set this up. 

Then, I pulled the ollama image on Docker within the EC2 instance, allowing me to run the container smoothly without limited RAM. 

Lastly, install Jupyter within SSH, configure the application to listen on all interfaces (0.0.0.0), set the port (I used 8889), and you can run Jupyter after running the following bash commands:

jupyter notebook --ip=0.0.0.0 --port=8889 --no-browser --allow-root
ssh -i /home/seanhegede/myenv/blank.pem -L 10006:localhost:8889 -L 11434:localhost:11434 ubuntu@18.xxx.xx.xx

To run the scraper, I opened a tmux shell and saved the code into the nano editor, where I could then compile the code directly in the terminal. An effective scraper should include real-time content handling,
hierarchical page navigation to search deep into subpages, block-based content parsing, and a failure management system to retry loading failed pages. You save the extracted embeddings into a .json file:
my script took 24 hours to compile. 

## Usage
When designing the RAG script, I used:

- Semantic similarity search: used all-MiniLM-L6-v2 from SentenceTransformer to encode both queries and documents
- Vector Index: Employs FAISS IndexFlatIP (Inner Product) for similarity search
- Normalization: L2-normalizes embeddings before indexing and querying for cosine similarity
- Relevance-based, light reranking: takes top-k results and filters out chunks below the similarity threshold (default 0.25)
- Top-K Limiting: Retrieves only the top 3 most similar chunks by default
- System prompt: instructing LLM to answer cohesively and as an acquisitions expert would

I loaded five real-estate related queries into my grid test script, and randomly selected 12 configurations from a matrix of different temperature, top-k, token size, min-similarity, and microstat 
parameters to run the queries on. 

The metrics I used to measure answer quality were: relevance, factual accuracy, coherence/ legibility, source usage, response length (word count), and hallucination (how likely model returned unsupported info). 
Besides response length, all were scaled from 0 (bad) - 1 (good). The quality metrics were also weighted to create an overall quality score: 
relevance 25%, accuracy 25%, coherence 20%, source usage 15%, hallucination resistance 15%. 

The configuration with the best overall results were: temperature = 0.3, top-p = 0.95, llm  top-k = 10, retrieval top-k = 8, min_similarity = 0.35, max_tokens = 200, microstat_eta = 0.5, 
microstat_tau = 5.0. (Default: temperature = 1.0, top-k = 64, top-p: 0.95)
## Limitations
In subsequent runs, I would like to beef up the RAG reranking to include cross-encoder reranking, employ more robust SentenceTransformer models like all-mpnet-base-v2 for better context
retrieval, and use a hybrid, lexical/keyword search approach to search for sources faster. However, these all come with a significant runtime trade-off. Perhaps increasing CPU and RAM volume in AWS 
could compensarte for this performance loss. 
