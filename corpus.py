import os
import sys
import tarfile
import urllib

from tqdm import tqdm

"""

Corpus : Polarity Dataset. Pang/Lee ACL 2004

http://www.cs.cornell.edu/people/pabo/movie-review-data/

"""
corpus_url = "http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz"

corpus_root = os.path.join(os.getcwd(), "review_polarity", "txt_sentoken")
catgeories = ["pos", "neg"]


# download corpus as a zip and then unzip
# downloads and unzips in the same directory
# by default set to current dir
def download_and_unzip():
    file_name = corpus_url.split("/")[-1]
    download_path = os.path.join(os.getcwd(), file_name)
    # where the zip will get extracted
    extracted_path = os.path.join(os.getcwd(), "review_polarity")

    if os.path.exists(extracted_path):
        print("Already downloaded and extracted!")
    else:
        # ============================================ download
        print("Downloading, sit tight!")

        def _progress(count, block_size, total_size):
            sys.stdout.write(
                f"\r>> Downloading {file_name} {float(count * block_size) / float(total_size) * 100.0}%")
            sys.stdout.flush()

        file_path, _ = urllib.request.urlretrieve(
            corpus_url, download_path, _progress)
        print()
        print(
            f"Successfully downloaded {file_name} {os.stat(file_path).st_size} bytes")

        # ======================================= unzip
        print()
        print("Unzipping ...")
        # create dir at extracted_path
        os.mkdir(extracted_path)
        tarfile.open(file_path, "r:gz").extractall(extracted_path)

        # =========================================== clean up
        # delete the downloaded zip file
        print("Deleting downloaded zip file")
        os.remove(file_path)


# just read all files for a category


def read_text_files(path):
    file_list = os.listdir(path)
    texts = []

    for fname in file_list:
        fpath = os.path.join(path, fname)

        f = open(fpath, mode="r")
        lines = f.read()
        texts.append(lines)
        f.close()

    return texts
