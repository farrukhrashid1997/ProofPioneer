import pandas as pd
from utils.gemini_interface import GeminiAPI
import multiprocessing
from tqdm import tqdm
import json

def initialize_api():
    """Initialize API instance for each worker process"""
    return GeminiAPI(
        model_name="gemini-1.5-flash-002", 
        secrets_file="./secrets/gemini_keys.json", 
        response_mime_type="application/json"
    )

def process_row(args):
    api = initialize_api()
    row, prompt_template = args
    prompt = prompt_template.replace("[Insert the claim here]", row['claim'])
    prompt = prompt.replace("[Insert the reasoning here]", row['text'])
    prompt = prompt.replace("[Insert the label here]", row['label'])
    response = api.get_llm_response(prompt, force_rotate=True)
    resp = json.loads(response.text) 
    row['justification'] = resp["Summary"]
    row['claim_author'] = resp["Claim Author"]
    row['claim_date'] = resp["Claim Date"]
    return row


def save_batch_to_csv(batch, output_file):
    pd.DataFrame(batch).to_csv(output_file, mode='w', header=True, index=False)

if __name__ == "__main__":
    df = pd.read_csv("snopes_results_cleaned.csv")
    
    try:
        existing_df = pd.read_csv("snopes_results_with_justification.csv")
        processed_claims = set(existing_df['claim'].tolist())
        df = df[~df['claim'].isin(processed_claims)]
        print(f"Found {len(processed_claims)} existing claims. Processing {len(df)} new claims.")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print("No existing justification file found. Processing all claims.")

    with open("prompts/justification_extraction.txt", "r") as f:
            justification_prompt_template = f.read()
            response_schema_justification = {
                "type": "string"
            }

    gemini_flash_api = GeminiAPI(model_name="gemini-1.5-flash-002", secrets_file="./secrets/gemini_keys.json", response_mime_type="application/json")

    # Only proceed if there are new claims to process
    if len(df) > 0:
        records = df.to_dict('records')
        results = []
        batch_size = 5

        pool_size = min(2, multiprocessing.cpu_count())
        pool = multiprocessing.Pool(processes=pool_size)
        process_args = [(record, justification_prompt_template) for record in records]
        try:
            for result in tqdm(pool.imap_unordered(process_row, process_args), total=len(records), desc="Processing rows"):
                results.append(result)
                if len(results) >= batch_size:
                    save_batch_to_csv(results, "snopes_results_with_justification.csv")
                    results = []

            if results:
                save_batch_to_csv(results, "snopes_results_with_justification.csv")

        except KeyboardInterrupt:
            print("\nProcessing interrupted by user.")
        except Exception as e:
            print(f"An error occurred during processing: {e}")
        finally:
            pool.close()
            pool.join()
    else:
        print("No new claims to process.")


    
    
    
    
    