import pandas as pd

def filter_wrong_answers(input_csv: str, output_csv: str) -> None:
    # Read CSV and drop any auto-generated index columns
    df = pd.read_csv(input_csv)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Filter mismatches
    wrong = df[df["model_answer"] != df["correct_answer"]]

    wrong.to_csv(output_csv, index=False)
# ----- test -----
if __name__ == "__main__":
    name="llama"
    filter_wrong_answers(f"/home/guo_chen2023/LLM-ToM-GenAI/sample_results/{name}_merged/{name}.csv",
    f"/home/guo_chen2023/LLM-ToM-GenAI/sample_results/{name}_merged/{name}_failures.csv")
