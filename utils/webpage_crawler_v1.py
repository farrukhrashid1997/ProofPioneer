import argparse
import os
from time import sleep
import trafilatura
from trafilatura.settings import DEFAULT_CONFIG
from trafilatura.meta import reset_caches
import sys

# Increase max file size to 50MB
DEFAULT_CONFIG.MAX_FILE_SIZE = 50000000

def get_page(url):
    page = None
    for i in range(3):
        try:
            page = trafilatura.fetch_url(url, config=DEFAULT_CONFIG) # Maybe use other crawlers to fetch the page. 
            # Args:
            # url: URL of the page to fetch.
            # no_ssl: Don't try to establish a secure connection (to prevent SSLError).
            # config: Pass configuration values for output control.
            # options: Extraction options (supersedes config).
            assert page is not None
            # print(f"Fetched {url}", file=sys.stderr)
            break
        except Exception as e:
            print(f"Attempt {i+1} failed for {url}: {str(e)}", file=sys.stderr)
            sleep(3)
    return page

def html2lines(page):
    if not page or len(page.strip()) == 0:
        return []

    # text = trafilatura.extract(page, config=DEFAULT_CONFIG)
    text = trafilatura.extract(page, favor_recall=True, include_tables=True, include_formatting=True, no_fallback=False)
    reset_caches()

    if text is None:
        return []

    return text.split("\n")

def url2lines(url):
    page = get_page(url)
    if page is None:
        return []
    return html2lines(page)

def save_content(url, output_dir):
    lines = url2lines(url)
    if not lines:
        print(f"No content extracted from {url}", file=sys.stderr)
        return

    # Create a filename from the URL
    filename = url.split("//")[-1].replace("/", "_")
    filename = os.path.join(output_dir, f"{filename}.txt")

    with open(filename, "w", encoding="utf-8") as f:
        f.write(url + "\n")  # Write URL as the first line
        f.write("\n".join(lines))

    print(f"Content saved to {filename}", file=sys.stderr)

# def main():
#     parser = argparse.ArgumentParser(description="Fetch and save webpage content.")
#     parser.add_argument("url", help="URL of the webpage to fetch")
#     parser.add_argument("--output_dir", default="webpage_content", help="Directory to save the content")
#     args = parser.parse_args()

#     if not os.path.exists(args.output_dir):
#         os.makedirs(args.output_dir)

#     save_content(args.url, args.output_dir)

# if __name__ == "__main__":
#     main()