"""
Fantom Dataset Loader

Load the Fantom v1 dataset from Google Cloud Storage.
Dataset URL: https://storage.googleapis.com/fantom_dataset/fantom_v1.json

Requirements:
    pip install google-cloud-storage
    OR use gcloud CLI with: gcloud auth application-default login
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FantomDataset:
    """Loader for Fantom v1 dataset from Google Cloud Storage."""
    
    DEFAULT_URL = "https://storage.googleapis.com/fantom_dataset/fantom_v1.json"
    
    def __init__(self, cache_dir: str = "./data"):
        """
        Initialize the dataset loader.
        
        Args:
            cache_dir: Directory to cache the downloaded dataset
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "fantom_v1.json"
        self.data: Optional[List[Dict[str, Any]]] = None
    
    def download(self, url: str = DEFAULT_URL, force: bool = False) -> Path:
        """
        Download the dataset from GCS using gsutil or google-cloud-storage.
        
        Args:
            url: URL to download from (gs:// or https://)
            force: If True, re-download even if cached file exists
            
        Returns:
            Path to the downloaded file
        """
        if self.cache_file.exists() and not force:
            logger.info(f"Using cached dataset at {self.cache_file}")
            return self.cache_file
        
        # Convert https:// to gs:// format
        gs_url = url.replace(
            "https://storage.googleapis.com/",
            "gs://"
        )
        
        logger.info(f"Downloading dataset from {gs_url}...")
        
        # Method 1: Try using gsutil (if gcloud CLI is installed)
        try:
            logger.info("Trying gsutil cp...")
            result = subprocess.run(
                ["gsutil", "cp", gs_url, str(self.cache_file)],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"Dataset downloaded and saved to {self.cache_file}")
            return self.cache_file
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"gsutil failed: {e}")
        
        # Method 2: Try using google-cloud-storage library
        try:
            logger.info("Trying google-cloud-storage library...")
            from google.cloud import storage
            
            # Parse bucket and blob from gs:// URL
            gs_path = gs_url.replace("gs://", "")
            bucket_name, blob_path = gs_path.split("/", 1)
            
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            
            blob.download_to_filename(str(self.cache_file))
            logger.info(f"Dataset downloaded and saved to {self.cache_file}")
            return self.cache_file
            
        except Exception as e:
            logger.error(f"google-cloud-storage failed: {e}")
        
        # Method 3: Try public access with requests library
        try:
            logger.info("Trying public HTTPS access...")
            import requests
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            with open(self.cache_file, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Dataset downloaded and saved to {self.cache_file}")
            return self.cache_file
            
        except Exception as e:
            logger.error(f"All download methods failed. Last error: {e}")
            raise RuntimeError(
                f"Failed to download dataset. Please try manually:\n"
                f"  gsutil cp {gs_url} {self.cache_file}\n"
                f"Or authenticate with: gcloud auth application-default login"
            )
    
    def load(self, url: str = DEFAULT_URL, force_download: bool = False) -> List[Dict[str, Any]]:
        """
        Load the dataset into memory.
        
        Args:
            url: URL to download from if not cached
            force_download: If True, re-download even if cached
            
        Returns:
            List of dataset entries
        """
        if self.data is not None and not force_download:
            logger.info("Using already loaded dataset from memory")
            return self.data
        
        # Download if needed
        self.download(url, force=force_download)
        
        # Load JSON
        logger.info(f"Loading dataset from {self.cache_file}...")
        with open(self.cache_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} entries")
        return self.data
    
    def __len__(self) -> int:
        """Return the number of entries in the dataset."""
        if self.data is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single entry by index."""
        if self.data is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        return self.data[idx]
    
    def __iter__(self):
        """Iterate over dataset entries."""
        if self.data is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        return iter(self.data)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get basic statistics about the dataset."""
        if self.data is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        stats = {
            "total_entries": len(self.data),
            "sample_keys": list(self.data[0].keys()) if self.data else [],
        }
        
        return stats


def load_fantom_dataset(
    cache_dir: str = "./data",
    force_download: bool = False
) -> List[Dict[str, Any]]:
    """
    Convenient function to load the Fantom dataset.
    
    Args:
        cache_dir: Directory to cache the dataset
        force_download: If True, re-download even if cached
        
    Returns:
        List of dataset entries
        
    Example:
        >>> data = load_fantom_dataset()
        >>> print(f"Loaded {len(data)} entries")
        >>> print(data[0])  # First entry
    """
    dataset = FantomDataset(cache_dir=cache_dir)
    return dataset.load(force_download=force_download)


if __name__ == "__main__":
    # Example usage
    print("Loading Fantom v1 dataset...")
    
    dataset = FantomDataset(cache_dir="./data")
    data = dataset.load()
    
    print(f"\n✓ Successfully loaded {len(data)} entries")
    print(f"\nDataset statistics:")
    stats = dataset.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    if data:
        print(f"\nFirst entry sample:")
        print(json.dumps(data[0], indent=2, ensure_ascii=False)[:500] + "...")
