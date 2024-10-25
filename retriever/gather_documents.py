import json
import os
from tqdm import tqdm
from multiprocessing import Manager, Lock
import multiprocessing
from utils.crawler import url2lines
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
from urllib.parse import urlparse


# Blacklists
BLACKLIST_DOMAINS = {
    "facebook.com",
    "huggingface.co",
}

def get_domain_name(url):
    if '://' not in url:
        url = 'http://' + url
    domain = urlparse(url).netloc
    return domain[4:] if domain.startswith("www.") else domain

def should_filter_link(link):
    domain = get_domain_name(link)
    if domain in BLACKLIST_DOMAINS:
        return True
    if link.endswith((".pdf", ".doc")):
        return True
    return False

def get_scraped_content(link, store_file_path):
    try:
        page_json = url2lines(link)
        page_json_str = json.dumps(page_json, ensure_ascii=False, indent=4)
        with open(store_file_path, "w", encoding="utf-8") as out_f:
            out_f.write(page_json_str)
        gc.collect()
        print(f"Successfully processed: {link}")
        return True, link, store_file_path
    except Exception as e:
        print(f"Error Processing: {e}")
        return False, link, store_file_path

def worker_task(args): 
    link, store_file_path = args
    return get_scraped_content(link, store_file_path)


def main():
    # Initialize multiprocessing manager and shared data structures
    manager = Manager()
    visited = manager.dict()
    lock = Lock()
    # Initialize variable
    arguments = set()
    store_counter = 0
    
    with open("outputs/search_results.json", "r") as fp:
        search_results = json.load(fp)
        document_folder = "outputs/documents"
        os.makedirs(document_folder, exist_ok=True)

    for claim_index, (claim, queries) in enumerate(tqdm(search_results.items())):
        for query_index, (query, page_results) in enumerate(queries.items()):
            for page_num, results in page_results.items():
                for webpage_index, result_object in enumerate(results):
                    link = str(result_object["link"]).strip()
                    if should_filter_link(link):
                        continue
                    webpage_key = f"{claim_index}-{query_index}-{page_num}-{webpage_index}"
                    
                    if link in visited:
                        store_file_path = visited[link]
                        
                    else:
                        store_counter +=1
                        store_file_path = os.path.join(
                            document_folder, f"webpage_{store_counter}.json"
                        )
                        visited[link] = store_file_path
                        arguments.add((link, store_file_path))
                        
    print(f"Total unique links to process: {len(arguments)}")
    # Get the number of cpus available - leaving one core free is important to leave
    # for other tasks
    cpu_count = multiprocessing.cpu_count() - 1
    print(f"Starting multiprocessing with {cpu_count} processes.")
    
    with ProcessPoolExecutor(max_workers=cpu_count) as executor:
        # Submit all unique futures with the response of the worker task as a key.
        future_to_args = {executor.submit(worker_task, arg): arg for arg in arguments}
        
        for future in tqdm(as_completed(future_to_args), total=len(future_to_args)):
            success, url, path = future.result()
            if not success:
                print(f"Failed to process link: {url}")
        

if __name__ == "__main__":
    main()