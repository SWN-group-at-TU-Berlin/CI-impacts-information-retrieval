# CI-impacts-information-retrieval
Feasibility study to test the ability of LLMs to extract information about critical infrastructure (CI) impacts from text sources.

For this small project, we aim to focus on text sources about the Ahr valley flooding from 2021. Mainly those sources mentioned in [Koks et al. 2022](https://doi.org/10.5194/nhess-22-3831-2022) will be used to test the applicability of the models. 
To conduct the information retrieval task a vector database is created storing all the text sources about the flood event. By using Retrieval Augmented Generation (RAG) these text sources are used by the LLM(s) to extract the different types of impacts of critical infrastructure failures, ranging from structural damages, over cascading effects to other CI sectors, to societal and economic impacts.

The `main` branch always contains an applicable version of the project, while feature-branches represent the current working states. Their branch history may change due to the use of `git rebase`.


## Getting started

All system requirements for the project are configured in a `pyproject.toml`.\
For package and dependency management we use the `uv` library. It is written in `Rust` and thus faster and more lightweight than Anaconda/Mamba or Poetry. Commands in `uv` are similar to the ones for `pip`. 

Set up the project simply by running 
````
uv sync
````
which creates a new virtual environment, stored in `.venv`\
Activate the environment by running
````
source .venv/bin/activate
````

### Generation of embeddings directly in Postgres DB

We aim to test different multilingual embedding models based on the suggestions from [HuggingFace Embedding Leaderboard](https://huggingface.co/spaces/mteb/leaderboard). However, we apply first open-source, leightweighted models, such as from Ollama, to setup the pipeline and for creating a benchmark.\
To facilitate the generation and management of the embedding vectors stored in the postgres DB, along with the original text sources, we are making use of the [`pgai` package](https://github.com/timescale/pgai/tree/main/docs). Its implementation enables to use a postgres DB efficiently for Retrieval-Augmented Generation (RAG). However, `pgai` should be executed within a docker container. We create the database in the container by making use of the[timescale DB](https://hub.docker.com/r/timescale/timescaledb), an open-source database designed to make SQL scalable for large, (time-series) data and complex queries. \
The creation of the database within the container and the creation of the embeddings by using a vectorizer, is done by following the steps described [here](https://github.com/timescale/pgai/blob/main/docs/vectorizer/quick-start.md). Make sure to have at least 15GB of free disk space and of course you need to have Docker installed.\
**Note**:  Download the [pgai extension](https://github.com/timescale/pgai/tree/main/projects/extension) via `git clone` in your project folder, before you you connect to the DB located in the docker container (see, step `docker compose exec -it db psql` mentioned [here](https://github.com/timescale/pgai/blob/main/docs/vectorizer/quick-start.md) )


## References

*Koks, E. E., van Ginkel, K. C. H., van Marle, M. J. E., and Lemnitzer, A.: Brief communication: Critical infrastructure impacts of the 2021 mid-July western European flood event, Nat. Hazards Earth Syst. Sci., 22, 3831–3838, https://doi.org/10.5194/nhess-22-3831-2022, 2022.*




