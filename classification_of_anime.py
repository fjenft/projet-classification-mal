import groq
import pandas as pd
import os
import pandas as pd
import json
from tqdm import tqdm
import time

# Creating a new dataframe with only the "Synopsis" column
df = pd.read_csv('anime-dataset-2023.csv')
Sdataframe = df[['Synopsis']]

# Display the first few rows of the new dataframe to verify
Sdataframe.head(10)

# Load the dataset
df = pd.read_csv('anime-dataset-2023.csv')

# Create a dataframe with only the "Synopsis" column
Sdataframe = df[['Synopsis']]

# Select the first 100 synopses
synopsis_dataframe = Sdataframe.head(100)

# Display the first few rows to verify
print(synopsis_dataframe.head())

# Function to process the DataFrame and return classification results
def process_synopsis_dataframe(dataframe, client, batch_size=10, retries=3):
    """
    Processes a dataframe containing anime synopses and classifies each synopsis into genres with scores.

    Args:
        dataframe (pd.DataFrame): The dataframe containing the synopsis column.
        client: The API client for generating classifications.
        batch_size (int): Number of synopses to process in each batch.
        retries (int): Number of retries in case of API errors.

    Returns:
        pd.DataFrame: Dataframe containing the classification results.
    """
    # Initialize results
    results = []

    # Process synopses in batches
    with tqdm(total=len(dataframe), desc="Processing synopses") as pbar:
        for index in range(0, len(dataframe), batch_size):
            batch = dataframe.iloc[index:index + batch_size]
            batch_synopses = batch['Synopsis'].tolist()

            # Retry classification if needed
            response = retry_classification(batch_synopses, index, client, retries)
            if response:
                results.extend(response)

            # Wait between API calls to avoid rate limits
            time.sleep(2)

            # Update progress bar
            pbar.update(len(batch))

    # Convert results to a dataframe and return
    return pd.DataFrame(results)

# Retry logic for handling API errors
def retry_classification(batch_synopses, batch_start_index, client, retries=3):
    for attempt in range(retries):
        try:
            response = classify_synopsis_group(batch_synopses, batch_start_index, client)
            if response:
                return response
        except Exception as e:
            print(f"Error during API call for batch starting at {batch_start_index + 1}: {e}")
        print(f"Retrying... Attempt {attempt + 1}/{retries}")
        time.sleep(3)  # Wait before retrying
    print(f"Failed to process batch starting at {batch_start_index + 1} after {retries} attempts.")
    return None

# Function to classify a group of synopses
def classify_synopsis_group(synopsis_batch, group_start_index, client):
    # Format the prompt
    synopses_in_prompt = "\n".join(
        [f"{group_start_index + i + 1}. {synopsis}" for i, synopsis in enumerate(synopsis_batch)]
    )
    prompt = f"""
    You are an expert in anime classification with an in-depth understanding of different genres such as 'Action', 'Adventure', 'Comedy', 'Drama', 'Fantasy', 'Horror', 'Mystery', 'Romance', 'Sci-Fi', and 'Supernatural'.

    You will be provided with a list of anime along with their synopses. For each anime, carefully analyze its synopsis and assign a score from 0 to 10 for each genre, where:
    - 0 means the genre is not applicable at all.
    - 10 means the genre is fully representative of the anime.

    Carefully consider the synopsis, and provide the most accurate scores for all genres. Each anime must have scores for **ALL genres** listed, even if some genres are scored 0.

    **Output the results STRICTLY in this JSON format:**
    [
      {{
        "anime_id": "Anime ID 1",
        "scores": {{
          "Action": Score1,
          "Adventure": Score2,
          "Comedy": Score3,
          "Drama": Score4,
          "Fantasy": Score5,
          "Horror": Score6,
          "Mystery": Score7,
          "Romance": Score8,
          "Sci-Fi": Score9,
          "Supernatural": Score10
        }}
      }},
      ...
    ]

    Synopses:
    {synopses_in_prompt}
    """

    # Send the request to the API
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[{"role": "system", "content": prompt}]
    )

    # Parse the response
    response_content = completion.choices[0].message.content.strip()
    if response_content.startswith("[") and response_content.endswith("]"):
        try:
            parsed_content = json.loads(response_content)
            for i, item in enumerate(parsed_content):
                item["anime_id"] = group_start_index + i + 1
            return parsed_content
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
            return None
    else:
        print("Invalid response format. Skipping batch.")
        return None

# Main Execution
if __name__ == "__main__":
    # Prompt the user for their API key
    api_key = input("Enter your API key: ").strip()
    if not api_key:
        raise ValueError("API key is required to proceed.")
    os.environ["GROQ_API_KEY"] = api_key

    # Initialize the client
    try:
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
    except Exception as e:
        print(f"Failed to initialize Groq client: {e}")
        exit()

    # Process the dataframe and get the results
    classified_results = process_synopsis_dataframe(synopsis_dataframe, client)

    # Display the results as a DataFrame
    print(classified_results)

    # Save results to a CSV file
    output_file = "classified_anime_results.csv"
    classified_results.to_csv(output_file, index=False)
    print(f"Results have been saved to {output_file}.")
