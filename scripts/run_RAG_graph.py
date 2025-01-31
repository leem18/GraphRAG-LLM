import os
import sys
from dotenv import load_dotenv

def set_environment_variables(env_file):
    load_dotenv(env_file)

def main():
    # Load API keys from key.env
    set_environment_variables('./config/key.env')
    set_environment_variables('./config/config.env')

    # Set API keys
    keys = [
        "OPENAI_API_KEY",
        "LANGCHAIN_TRACING_V2",
        "TAVILY_API_KEY",
        "GOOGLE_CSE_ID",
        "GOOGLE_API_KEY",
        "SERPAPI_KEY",
        "NAVER_CLIENT_ID",
        "NAVER_CLIENT_SECRET",
        "LANGCHAIN_ENDPOINT",
        "LANGCHAIN_API_KEY",
        "LANGCHAIN_PROJECT"
    ]
    for key in keys:
        os.environ[key] = os.getenv(key)

    os.environ['USER_AGENT'] = 'myagent'

    # Run the GAN_lg.py script
    os.system('python3 ./rag_graph_pipeline.py')

if __name__ == "__main__":
    main()
