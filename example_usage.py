"""
Example: How to use the Fantom dataset loader
"""

from fantom import load_fantom_dataset, FantomDataset
import json


def example_simple_load():
    """Simple way to load the dataset."""
    print("=" * 60)
    print("Example 1: Simple loading")
    print("=" * 60)
    
    data = load_fantom_dataset()
    print(f"Loaded {len(data)} entries")
    print(f"\nFirst entry keys: {list(data[0].keys())}")
    

def example_class_usage():
    """Using the FantomDataset class."""
    print("\n" + "=" * 60)
    print("Example 2: Using FantomDataset class")
    print("=" * 60)
    
    dataset = FantomDataset(cache_dir="./data")
    data = dataset.load()
    
    # Get statistics
    stats = dataset.get_statistics()
    print(f"\nDataset statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Access by index
    print(f"\nEntry 0 set_id: {dataset[0]['set_id']}")
    print(f"Entry 0 conv_id: {dataset[0]['conv_id']}")
    
    # Iterate
    print(f"\nFirst 3 conversation IDs:")
    for i, entry in enumerate(dataset):
        if i >= 3:
            break
        print(f"  {i}: {entry['conv_id']}")


def example_explore_data():
    """Explore the dataset structure."""
    print("\n" + "=" * 60)
    print("Example 3: Exploring data structure")
    print("=" * 60)
    
    data = load_fantom_dataset()
    sample = data[0]
    
    print(f"\nSample entry structure:")
    print(f"  set_id: {sample['set_id']}")
    print(f"  part_id: {sample['part_id']}")
    print(f"  conv_id: {sample['conv_id']}")
    print(f"  full_context length: {len(sample['full_context'])} chars")
    print(f"  short_context length: {len(sample['short_context'])} chars")
    print(f"  missed_info: {sample['missed_info'][:100]}...")
    print(f"  joining_speaker: {sample['joining_speaker']}")
    
    # Count different types of QAs
    print(f"\n  factQA: {type(sample.get('factQA'))}")
    print(f"  Number of beliefQAs: {len(sample.get('beliefQAs', []))}")
    print(f"  Number of infoAccessibilityQA_list: {len(sample.get('infoAccessibilityQA_list', []))}")
    print(f"  Number of answerabilityQA_list: {len(sample.get('answerabilityQA_list', []))}")
    
    # Show a QA example
    if sample.get('factQA'):
        print(f"\n  Sample factQA:")
        qa = sample['factQA']  # factQA is a dict, not a list
        print(f"    Question: {qa.get('question', 'N/A')}")
        print(f"    Answer: {qa.get('correct_answer', 'N/A')[:100]}...")
        print(f"    Type: {qa.get('question_type', 'N/A')}")


def example_filter_data():
    """Filter dataset entries."""
    print("\n" + "=" * 60)
    print("Example 4: Filtering data")
    print("=" * 60)
    
    data = load_fantom_dataset()
    
    # Filter by conversation ID
    conv_0_entries = [e for e in data if e['conv_id'] == 0]
    print(f"Entries with conv_id=0: {len(conv_0_entries)}")
    
    # Count unique conversation IDs
    unique_convs = set(e['conv_id'] for e in data)
    print(f"Unique conversation IDs: {len(unique_convs)}")
    
    # Find entries with many belief QAs
    many_beliefs = [e for e in data if len(e.get('beliefQAs', [])) > 5]
    print(f"Entries with >5 beliefQAs: {len(many_beliefs)}")


if __name__ == "__main__":
    example_simple_load()
    example_class_usage()
    example_explore_data()
    example_filter_data()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
