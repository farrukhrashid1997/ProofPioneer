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

def process_row(row, prompt_template):
    """Process a single row, handling errors gracefully"""
    
    # Initialize new fields with None
    row['justification'] = None
    row['claim_author'] = None
    row['claim_date'] = None
    try:
        api = initialize_api()
        prompt = prompt_template.replace("[Insert the claim here]", row['claim'])
        prompt = prompt.replace("[Insert the reasoning here]", row['text'])
        prompt = prompt.replace("[Insert the label here]", row['label'])
        
        response = api.get_llm_response(prompt, force_rotate=True)
        if response is None:
            print(f"No response for claim: {row['claim'][:50]}...")
            return row
            
        resp = json.loads(response.text)
        row['justification'] = resp["Summary"]
        row['claim_author'] = resp["Claim Author"]
        row['claim_date'] = resp["Claim Date"]
        return row
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error for claim: {row['claim'][:50]}... Error: {e}")
        return row
        
    except Exception as e:
        print(f"Error processing claim: {row['claim'][:50]}... Error: {e}")
        return row


def save_batch_to_csv(batch, output_file):
    df_batch = pd.DataFrame(batch)
    try:
        # Check if file exists
        pd.read_csv(output_file)
        # File exists, append without headers
        df_batch.to_csv(output_file, mode='a', header=False, index=False)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        # File doesn't exist or is empty, create with headers
        df_batch.to_csv(output_file, mode='w', header=True, index=False)
        
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

        # pool_size = min(2, multiprocessing.cpu_count())
        # pool = multiprocessing.Pool(processes=pool_size)
        # process_args = [(record, justification_prompt_template) for record in records]
        try:
            for record in tqdm(records, desc="Processing rows"):
                result = process_row(record, justification_prompt_template)
                if len(results) >= batch_size:
                    save_batch_to_csv(results, "snopes_results_with_justification.csv")
                    results = []

            if results:
                save_batch_to_csv(results, "snopes_results_with_justification.csv")

        except KeyboardInterrupt:
            print("\nProcessing interrupted by user.")
        except Exception as e:
            print(f"An error occurred during processing: {e}")
        # finally:
        #     pool.close()
        #     pool.join()
    else:
        print("No new claims to process.")


    
    
    
    
    