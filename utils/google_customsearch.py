from googleapiclient.discovery import build
from time import sleep
import json
from itertools import cycle

class GoogleCustomSearch:
    def __init__(self, max_gs_api_calls=100, n_pages=1):
        self.max_gs_api_calls = max_gs_api_calls
        self.gs_api_counter = 0
        self.gs_secrets = None
        self.keys_pool = self._load_secrets("secrets/google_secrets.json")
        self.n_pages = n_pages

    def _load_secrets(self, secrets_file):
        with open(secrets_file) as fp:
            gs_api_secrets = json.load(fp)
        return cycle(gs_api_secrets)

    def _rotate_secret(self):
        current_secret = next(self.keys_pool)
        self.gs_secrets = current_secret
        self.gs_api_counter = 0
        print("Changing Google custom search SECRET")

    def _google_search(self, search_term, **kwargs):
        service = build("customsearch", "v1", developerKey=self.gs_secrets["api_key"])
        res = service.cse().list(q=search_term, cx=self.gs_secrets["cx"], **kwargs).execute()

        if "items" in res:
            return res['items']
        else:
            return []

    def _get_google_search_results(self, sort_date, search_string, page=0):
        search_results = []
        for attempt in range(3):
            try:
                search_results += self._google_search(
                    search_string,
                    num=10,
                    start=0 + 10 * page,
                    sort=f"date:r:19000101:{sort_date}",
                    dateRestrict=None,
                    gl="US"
                )
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
            if self.gs_api_counter > self.max_gs_api_calls or not self.gs_secrets:
                self._rotate_secret()

            page_results = self._get_google_search_results(sort_date, search_string, page=page_num)
            search_results[page_num+1] = page_results
            self.gs_api_counter += 1

        return search_results


# if __name__ == "__main__":
#     gcs = GoogleCustomSearch()
#     string = "President Joe Biden gave Congress an exemption from vaccine mandate"
#     results = gcs.fetch_results(string, "20220122")

#     with open("test.json", "w") as fp:
#         json.dump(results, fp, indent=4)