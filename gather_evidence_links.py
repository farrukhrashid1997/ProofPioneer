import json
import os
from nltk import pos_tag, word_tokenize
from tqdm import tqdm
import pandas as pd
from utils.gemini_interface import GeminiAPI
from utils.google_customsearch import GoogleCustomSearch
from utils.bing_customsearch import BingCustomSearch
import datetime
import random


def string_to_search_query(text, author):
    parts = word_tokenize(text.strip())
    tags = pos_tag(parts)
    keep_tags = ["CD", "JJ", "NN", "VB"] # Cardinal Numbers, Adjectives, Nouns, Verbs

    if author is not None:
        search_string = author.split()
    else:
        search_string = []

    for token, tag in zip(parts, tags):
        for keep_tag in keep_tags:
            if tag[1].startswith(keep_tag):
                search_string.append(token)

    search_string = " ".join(search_string)
    return search_string


# Write a function to create an folder called outputs
def create_output_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def parse_llm_questions_response(response):
    questions = []
    for sentence in response.text.split('\n'):
        sentence = sentence.strip()
        if sentence.endswith('?'):
            questions.append(sentence)
    return questions
 

def extract_and_format_date(check_date, default_date="2022-01-01"):

    # If the date is not provided, use the default date
    if check_date:
        year, month, date = check_date.split("-")
    else:
        year, month, date = default_date.split("-")
    
    if len(year) == 2 and int(year) <= 30:
        year = "20" + year
    elif len(year) == 2:
        year = "19" + year
    elif len(year) == 1:
        year = "200" + year

    month = month.zfill(2)
    date = date.zfill(2)

    return f"{year}{month}{date}"



if __name__ == "__main__":
    # Load the multilingual claims dataset
    
    with open("claim_datasets/averitec/train.json") as fp:
        claims_to_process = json.load(fp) 
        claims_to_process = random.sample(claims_to_process, 10) # Pick 50 random examples for testing
        min_date = "2022-01-01" # YYYY-MM-DD
    # rows_with_claimDate = claims_df[claims_df["claimDate"] != "none"]     
    # claims_to_process = rows_with_claimDate.sample(10)
    # min_date = "2020-01-01" # YYYY-MM-DD
    
    # # Question generation prompt
    with open("prompts/prompt_2Q.txt", "r") as f:
        question_prompt_template = f.read()

    # # Save folder
    save_folder = "outputs"
    create_output_folder(save_folder)

    # Google Search
    n_pages = 1
    max_api_calls_per_account = 100
    google_search = GoogleCustomSearch(max_api_calls_per_account, n_pages)
    # bing_search = BingCustomSearch(max_api_calls_per_account, n_pages)

    # # Gemini Flash 1.5 Interface 
    gemini_flash_api = GeminiAPI(model_name="gemini-1.5-flash-latest")
    # # gemini_pro_api = GeminiAPI(model_name="gemini-1.5-pro-latest")

    results = {}
    n_pages = 1
    claim_queries = {}

    # # Create a CSV file to store the search results
    results_filename = f"{save_folder}/search_results.json"
    claim_queries_filename = f"{save_folder}/claim_queries.json"

    existing = {}

    for index, ind_claim in tqdm(enumerate(claims_to_process)):
        claim = ind_claim["claim"]
        # Generate questions using Gemini API
        prompt = question_prompt_template.replace("[Insert the claim here]", claim)

        response = gemini_flash_api.get_llm_response(prompt)
        # # Extract the questions from the response
        llm_questions = parse_llm_questions_response(response)
        # Extract and format the date``
        sort_date = extract_and_format_date(ind_claim["claim_date"], default_date=min_date)
        search_strings = []
        search_types = []


        search_string = string_to_search_query(claim, None)
        search_strings.append(search_string)
        search_types.append("claim")

    # 3. Process and remove duplicate questions (if any)
        processed_questions = set()
        for question in llm_questions:
            processed_question = string_to_search_query(question, None)
            if processed_question not in search_strings:
                processed_questions.add(processed_question)
                search_strings.append(processed_question)
                search_types.append("generated_question")

        search_results = []
        claim_queries[claim] = []
        
        store_counter = 0
        ts = []

        print("PROCESSING CLAIM: ", claim)
        results[claim] = {}

        for this_search_string, this_search_type in zip(search_strings, search_types):
    #         # Bookkeeping
            claim_queries[claim].append((
                this_search_string,
                this_search_type
            ))

            sstring_search_results = google_search.fetch_results(this_search_string, sort_date)
            results[claim][this_search_string] = sstring_search_results

        # Save updated results and claim_queries after processing each claim
        with open(results_filename, "w", encoding='utf-8') as fp:
            json.dump(results, fp, indent=4, ensure_ascii=False)

        with open(claim_queries_filename, "w", encoding='utf-8') as fp:
            json.dump(claim_queries, fp, indent=4, ensure_ascii=False)


# # TODO: - How do you know that the questions generated are more relevant? (Sir Saqib) 
# # How much evidence is required to support or refute the claim?
# # How would you remove the noise?
# # What if there is no evidence available? (maybe a fallback?)
# # Refer techniques from different papers - 
# # Try out different embedding models - cosine similarity, jaccard similarity, 
