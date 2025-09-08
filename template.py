import os
from pathlib import Path
import logging
from logging import Logger

logging.basicConfig(level=logging.INFO)
logger: Logger = logging.getLogger(__name__)

project_name = "Breast_Cancer_Detection"

file_paths=[
    "src",
    f"src/{project_name}",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transfer.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/data_preprocessing.py",
    f"src/{project_name}/pipeline/model_training.py",
    f"src/{project_name}/pipeline/model_evaluation.py",
    f"src/{project_name}/config.py",
    f"src/{project_name}/main.py",
    f"configs/config.yaml",
    f"src/exceptions.py",
    f"src/utils.py",
    f"src/logger.py",
    "main.py",
    "artifacts",
    "notebooks",
    "requirements.txt",
    "README.md",
    "setup.py",
    "Dockerfile",
    ".dockerignore",
    ".gitignore",
    ".env"
    
]

for filepath in file_paths:
    filepath=Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Created directory: {filedir}")
        
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, "w") as f:
            pass
        logging.info(f"Created file: {filepath}")
    else:
        logging.info(f"File already exists: {filepath}")
        
    