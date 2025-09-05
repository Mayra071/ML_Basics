import os
from pathlib import Path
import logging
from logging import Logger

logging.basicConfig(level=logging.INFO)
logger: Logger = logging.getLogger(__name__)

project_name = "DiabetesPrediction"

file_paths=[
    # f"src/{project_name}",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transfer.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_monitering.py",
    
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/data_preprocessing.py",
    f"src/{project_name}/pipelines/model_training.py",
    f"src/{project_name}/pipelines/model_evaluation.py",
    
    
    f"src/{project_name}/exception.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/utils.py",
    "app.py",
    # "requirements.txt",
    "README.md",
    # "setup.py",
    "Dockerfile",
    ".gitignore"
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
        
    