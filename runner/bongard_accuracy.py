import pandas as pd

class BongardAccuracy:
    """Simple class to evaluate accuracy for Bongard problem results."""

    def __init__(self, results_file: str):
        """Initialize with the path to the results CSV file."""
        self.results_file = results_file
        self.df = pd.read_csv(results_file)

    def calculate_accuracy(self):
        """Calculate and print the basic accuracy metrics."""
        total_records = len(self.df)

        # Create masks for correct predictions (reuse these for efficiency)
        mask_a_correct = (self.df["query_id"] == "A") & (self.df["predicted_answer"] == "positive")
        mask_b_correct = (self.df["query_id"] == "B") & (self.df["predicted_answer"] == "negative")

        # Get totals by query type (also reuse these)
        mask_a = self.df["query_id"] == "A"
        mask_b = self.df["query_id"] == "B"
        query_a_total = mask_a.sum()
        query_b_total = mask_b.sum()

        # Calculate overall accuracy
        query_a_correct = mask_a_correct.sum()
        query_b_correct = mask_b_correct.sum()
        correct_predictions = query_a_correct + query_b_correct
        accuracy = correct_predictions / total_records * 100

        # Calculate query-specific accuracy percentages
        query_a_accuracy = query_a_correct / query_a_total * 100
        query_b_accuracy = query_b_correct / query_b_total * 100

        # Print results in the exact format requested
        print(f"Total records: {total_records}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.2f}%")
        print()
        print(
            f"Query ID 'A' accuracy: {query_a_correct}/{query_a_total} "
            f"({query_a_accuracy:.2f}%)"
        )
        print(
            f"Query ID 'B' accuracy: {query_b_correct}/{query_b_total} "
            f"({query_b_accuracy:.2f}%)"
        )


if __name__ == "__main__":
    evaluator = BongardAccuracy(".csv")
    evaluator.calculate_accuracy()
