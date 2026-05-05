"""
Quick Runner: Execute the entire PII detection pipeline
Run this to execute all steps in the correct order with nice progress output.
"""

import subprocess
import sys
from pathlib import Path
import json

def run_command(description, command):
    """Run a command and report success/failure."""
    print("\n" + "=" * 80)
    print(f"📍 {description}")
    print("=" * 80)
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=False
        )
        print(f"✓ {description} completed successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with exit code {e.returncode}")
        print(f"  Command: {command}")
        return False


def print_summary(data_dir):
    """Print a summary of results."""
    print("\n" + "=" * 80)
    print("📊 PIPELINE COMPLETE - RESULTS SUMMARY")
    print("=" * 80)
    
    # Check what files were created
    files_created = []
    expected_files = [
        ("train_bio.jsonl", "Training data prepared"),
        ("val_bio.jsonl", "Validation data prepared"),
        ("test_bio.jsonl", "Test data prepared"),
        ("spacy_metrics.json", "SpaCy baseline evaluated"),
        ("spacy_predictions.jsonl", "SpaCy predictions saved"),
        ("bert_model", "BERT model trained"),
        ("bert_metrics.json", "BERT evaluated"),
        ("bert_predictions.jsonl", "BERT predictions saved"),
        ("model_comparison.json", "Models compared"),
    ]
    
    print("\n📁 Output Files:")
    print("-" * 80)
    for filename, description in expected_files:
        file_path = data_dir / filename
        if file_path.exists():
            print(f"  ✓ {filename:<30} {description}")
        else:
            print(f"  ✗ {filename:<30} (not created yet)")
    
    # Load and display metrics if available
    spacy_metrics_file = data_dir / "spacy_metrics.json"
    bert_metrics_file = data_dir / "bert_metrics.json"
    
    if spacy_metrics_file.exists() and bert_metrics_file.exists():
        print("\n" + "-" * 80)
        print("📈 Performance Comparison:")
        print("-" * 80)
        
        with open(spacy_metrics_file) as f:
            spacy_metrics = json.load(f)
        with open(bert_metrics_file) as f:
            bert_metrics = json.load(f)
        
        print(f"{'Metric':<15} {'SpaCy':<15} {'BERT':<15} {'Difference':<15}")
        print("-" * 80)
        
        for metric in ["accuracy", "precision", "recall", "f1"]:
            spacy_val = spacy_metrics.get(metric, 0)
            bert_val = bert_metrics.get(metric, 0)
            diff = bert_val - spacy_val
            
            diff_str = f"{diff:+.4f}"
            print(f"{metric:<15} {spacy_val:<15.4f} {bert_val:<15.4f} {diff_str:<15}")
    
    print("-" * 80)
    print("\n✓ Pipeline complete! Next steps:")
    print("  1. Review the results in model_comparison.json")
    print("  2. Check individual predictions in bert_predictions.jsonl and spacy_predictions.jsonl")
    print("  3. Read PIPELINE_GUIDE.md for detailed explanation of each step")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    data_dir = Path("/Users/evwu/Documents/Repositories/ml_project_PI_redaction")
    
    print("\n" + "🚀 " * 20)
    print("PII REDACTION PIPELINE - FULL RUN")
    print("🚀 " * 20)
    print("\nThis script will run 4 steps:")
    print("  1. Prepare data for BERT training (BIO format)")
    print("  2. Get SpaCy baseline performance")
    print("  3. Train BERT on your PII data (5-15 minutes)")
    print("  4. Compare BERT vs SpaCy results")
    
    response = input("\nContinue? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        sys.exit(0)
    
    # Step 1: Data preparation
    success = run_command(
        "Step 1/4: Preparing data for BERT",
        f"cd {data_dir} && python data_preparation.py"
    )
    if not success:
        print("Cannot continue without prepared data.")
        sys.exit(1)
    
    # Step 2: SpaCy baseline
    success = run_command(
        "Step 2/4: Running SpaCy baseline",
        f"cd {data_dir} && python spacy_baseline.py"
    )
    if not success:
        print("Warning: SpaCy baseline failed. Continuing anyway...")
    
    # Step 3: BERT training
    print("\n⏱️  Step 3/4: Training BERT (this will take several minutes...)")
    success = run_command(
        "Training BERT model",
        f"cd {data_dir} && python bert_training.py"
    )
    if not success:
        print("BERT training failed. Check error messages above.")
        sys.exit(1)
    
    # Step 4: BERT inference and comparison
    success = run_command(
        "Step 4/4: Running BERT inference and comparison",
        f"cd {data_dir} && python bert_inference.py"
    )
    if not success:
        print("BERT inference failed.")
        sys.exit(1)
    
    # Print summary
    print_summary(data_dir)
