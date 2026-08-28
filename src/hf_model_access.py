import os
from pathlib import Path
from typing import Optional

os.environ["HF_HOME"] = (
    "/beegfs/home/users/a/a-buch/_PROJECTS/CI-impacts-information-retrieval/notebooks/huggingface_mirror/hub/"
    # "/home/a-buch/Documents/TUB_DWN/_PROJECTS/CI-impacts-information-retrieval/notebooks/huggingface_mirror/"
)
## or set in ..bashrc 
## export HF_HOME="/beegfs/home/users/a/a-buch/_PROJECTS/CI-impacts-information-retrieval/notebooks/huggingface_mirror/hub/"
HF_HOME = os.environ["HF_HOME"]


def get_weight_dir(
    model_ref: str,
    *,
    model_dir: str = HF_HOME,
    revision: str = "main",
) -> Path:
    """
    Parse model name to locally stored weights.
    Args:
        model_ref (str) : Model reference containing org_name/model_name such as 'meta-llama/Llama-2-7b-chat-hf'.
        revision (str): Model revision branch. Defaults to 'main'.
        model_dir (str | os.PathLike[Any]): Path to directory where models are stored. Defaults to value of $HF_HOME (or present directory)

    Returns:
        str: path to model weights within model directory
    """
    model_dir = Path(model_dir)
    assert model_dir.is_dir()
    model_path = model_dir / "--".join(["models", *model_ref.split("/")])
    assert model_path.is_dir()
    snapshot_hash = (model_path / "refs" / revision).read_text()
    weight_dir = model_path / "snapshots" / snapshot_hash
    assert weight_dir.is_dir()
    return weight_dir
    
