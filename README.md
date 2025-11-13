# LLM-ToM-GenAI

Load and work with the Fantom v1 dataset from Google Cloud Storage.

## Dataset

- **URL**: `https://storage.googleapis.com/fantom_dataset/fantom_v1.json`
- **Size**: 870 entries
- **Description**: Theory of Mind (ToM) dataset with conversational contexts and QA pairs

## Installation

```bash
# Clone the repository
git clone https://github.com/jiayiichen/LLM-ToM-GenAI.git
cd LLM-ToM-GenAI

# Install dependencies
pip install -r requirements.txt

# Authenticate with Google Cloud (if needed)
gcloud auth application-default login
```

## Quick Start

### Simple Usage

```python
from fantom import load_fantom_dataset

# Load the dataset
data = load_fantom_dataset()
print(f"Loaded {len(data)} entries")

# Access an entry
print(data[0]['full_context'])
print(data[0]['factQA'])
```

### Using the Dataset Class

```python
from fantom import FantomDataset

dataset = FantomDataset(cache_dir="./data")
data = dataset.load()

# Get statistics
stats = dataset.get_statistics()
print(stats)

# Access by index
entry = dataset[0]
print(entry['set_id'])

# Iterate
for entry in dataset:
    print(entry['conv_id'])
```

## Dataset Structure

Each entry contains:

- `set_id`: Unique identifier for the set
- `part_id`: Part identifier
- `conv_id`: Conversation ID
- `full_context`: Complete conversation text
- `short_context`: Abbreviated context
- `missed_info`: Information missed by joining speaker
- `joining_speaker`: Name of speaker who joined mid-conversation
- `factQA`: Factual question-answer pair (dict)
- `beliefQAs`: Belief-based QA pairs (list)
- `infoAccessibilityQA_list`: Information accessibility questions (list)
- `answerabilityQA_list`: Answerability questions (list)
- `infoAccessibilityQAs_binary`: Binary accessibility QAs
- `answerabilityQAs_binary`: Binary answerability QAs

## Examples

Run the examples:

```bash
python3 fantom.py          # Basic loading and statistics
python3 example_usage.py   # Comprehensive examples
```

## Requirements

- Python 3.9+
- `google-cloud-storage` (for GCS access)
- `requests` (fallback download method)
- Google Cloud CLI with authentication (recommended)

## Troubleshooting

If download fails:

1. **Authenticate with gcloud**:
   ```bash
   gcloud auth application-default login
   ```

2. **Manual download**:
   ```bash
   gsutil cp gs://fantom_dataset/fantom_v1.json ./data/
   ```

3. **Check permissions**: Ensure your Google account has access to the bucket

## License

See dataset license at the source.

