import requests
import json
from itertools import cycle
from time import sleep

class BingCustomSearch:
    def __init__(self, max_bing_api_calls=100, n_pages=1):
        self.max_bing_api_calls = max_bing_api_calls
        self.bing_api_counter = 0
        self.bing_secrets = None
        self.keys_pool = self._load_secrets("secrets/bing_secrets.json")
        self.n_pages = n_pages

    def _load_secrets(self, secrets_file):
        with open(secrets_file) as fp:
            bing_api_secrets = json.load(fp)
        return cycle(bing_api_secrets)

    def _rotate_secret(self):
        current_secret = next(self.keys_pool)
        self.bing_secrets = current_secret
        self.bing_api_counter = 0
        print("Changing Bing custom search SECRET")

    def _bing_search(self, search_term, **kwargs):
        headers = {"Ocp-Apim-Subscription-Key": self.bing_secrets["api_key"]}
        params = {"q": search_term, "count": 10,  "customconfig": self.bing_secrets["customconfig_id"], **kwargs}
        response = requests.get(self.bing_secrets["endpoint"], headers=headers, params=params)
        response.raise_for_status()
        return response.json().get("webPages", {}).get("value", [])

    def _get_bing_search_results(self, sort_date, search_string, page=0):
        search_results = []
        for attempt in range(3):
            try:
                search_results += self._bing_search(
                    search_string,
                    offset=page * 10,
                    sort=f"date:{sort_date}",
                )
                sleep(3)
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                sleep(3)

        if not search_results:
            print(f"No results found for search string: {search_string}")

        return search_results

    def fetch_results(self, search_string, sort_date):
        search_results = {}
        for page_num in range(self.n_pages):
            if self.bing_api_counter > self.max_bing_api_calls or not self.bing_secrets:
                self._rotate_secret()

            page_results = self._get_bing_search_results(sort_date, search_string, page=page_num)
            search_results[page_num + 1] = page_results
            self.bing_api_counter += 1

        return search_results


if __name__ == "__main__":
    bcs = BingCustomSearch()
    string = "President Joe Biden gave Congress an exemption from vaccine mandate"
    results = bcs.fetch_results(string, "2022-01-22")

    with open("bing_results.json", "w") as fp:
        json.dump(results, fp, indent=4)