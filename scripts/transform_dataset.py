import json
import constants

def transform_records(data):
    output = []
    for record in data:
        uid = record["uid"]

        # Extract and prefix paths
        image_files = [
            f"{constants.DATA_DIR}/{img}" for img in record["imageFiles"]
        ]

        positive_set = [image_files[i] for i in range(0, 6)]
        negative_set = [image_files[i] for i in range(7, 13)]
        query_image_A = image_files[6]   # index 6
        query_image_B = image_files[13]  # index 13

        # A record
        record_A = {
            "uid": f"{uid}_A",
            "commonSense": record["commonSense"],
            "concept": record["concept"],
            "caption": record["caption"],
            "positive_set": positive_set,
            "negative_set": negative_set,
            "query_image": query_image_A
        }

        # B record
        record_B = {
            "uid": f"{uid}_B",
            "commonSense": record["commonSense"],
            "concept": record["concept"],
            "caption": record["caption"],
            "positive_set": positive_set,
            "negative_set": negative_set,
            "query_image": query_image_B
        }

        output.extend([record_A, record_B])
    return output

if __name__ == "__main__":

  # Read the image files metadata
  with open(constants.TEST_DATASET, 'r') as f:
    data = json.load(f)

  # Transform the records
  transformed_data = transform_records(data)

  # Save the transformed data
  with open(constants.TRANSFORMED_DATASET, 'w') as f:
    json.dump(transformed_data, f, indent=2)
