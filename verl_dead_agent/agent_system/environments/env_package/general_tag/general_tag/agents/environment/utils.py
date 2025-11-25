import os, re
from os.path import join as pjoin

import shutil
import tempfile
from os.path import join as pjoin

import requests
from tqdm import tqdm


def mkdirs(dirpath: str) -> str:
    """Create a directory and all its parents.

    If the folder already exists, its path is returned without raising any exceptions.

    Arguments:
        dirpath: Path where a folder need to be created.

    Returns:
        Path to the (created) folder.
    """
    try:
        os.makedirs(dirpath)
    except FileExistsError:
        pass

    return dirpath

def download(url, dst, desc=None, force=False):
    """Download a remote file using HTTP get request.

    Args:
        url (str): URL where to get the file.
        dst (str): Destination folder where to save the file.
        force (bool, optional):
            Download again if it exists]. Defaults to False.

    Returns:
        str: Path to the downloaded file.

    Notes:
        This code is inspired by
        https://github.com/huggingface/transformers/blob/v4.0.0/src/transformers/file_utils.py#L1069
    """
    filename = url.split("/")[-1]
    path = pjoin(mkdirs(dst), filename)

    if os.path.isfile(path) and not force:
        return path

    # Download to a temp folder first to avoid corrupting the cache
    # with incomplete downloads.
    temp_dir = mkdirs(pjoin(tempfile.gettempdir(), "tales"))
    temp_path = pjoin(temp_dir, filename)
    with open(temp_path, "ab") as temp_file:
        headers = {}
        resume_size = temp_file.tell()
        if resume_size:
            headers["Range"] = f"bytes={resume_size}-"
            headers["x-ms-version"] = "2020-04-08"  # Needed for Range support.

        r = requests.get(url, stream=True, headers=headers)
        if r.headers.get("x-ms-error-code") == "InvalidRange" and r.headers[
            "Content-Range"
        ].rsplit("/", 1)[-1] == str(resume_size):
            shutil.move(temp_path, path)
            return path

        r.raise_for_status()  # Bad request.
        content_length = r.headers.get("Content-Length")
        total = resume_size + int(content_length)
        pbar = tqdm(
            unit="B",
            initial=resume_size,
            unit_scale=True,
            total=total,
            desc=desc or "Downloading {}".format(filename),
            leave=False,
        )

        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                pbar.update(len(chunk))
                temp_file.write(chunk)

    shutil.move(temp_path, path)

    pbar.close()
    return path



# =======================================
# Helper functions for processing / cleaning obs and data
# =======================================
RE_TW_HEADER = re.compile(r"^\s*-=.*=-\s*$")
RE_TW_TASK   = re.compile(r"^\s*Your task is to:.*$")

def clean_cookingworld_obs(text: str) -> str:
    """
    Simple function to clean / trim the observation
    """
    # Start by cleaning the ascii art
    m = re.search(r"\${6,}(?![\s\S]*\${6,})([\s\S]*)\Z", text)
    if not m:
        text = text
    else:
        out = m.group(1)
        text = re.sub(r"^[ \t\r\n]+", "", out)
    
    # Now clean for other things
    cleaned = re.sub(r"-=\s*.*?\s*=-", "", text)   # Remove the 'location' flags in main obs
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)   # Reduce triple newlines to just double newlines
    cleaned = cleaned.strip("\n")                  # Remove trailing newlines
    return cleaned

def clean_alfworld_obs(text: str) -> str:
    cleaned = re.sub(r"-=\s*.*?\s*=-", "", text)   # Remove the 'location' flags in main obs
    cleaned = cleaned.strip("\n")                  # Remove trailing newlines
    return cleaned

def clean_alfworld_obs_notask(text: str) -> str:
    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if RE_TW_HEADER.match(stripped):
            continue
        if RE_TW_TASK.match(stripped):
            continue
        cleaned_lines.append(line)

    # Remove leading/trailing blank lines (newlines)
    return "\n".join(cleaned_lines).strip("\n")

# =======================
# Context Managers
# =======================

NECESSARY_CONTEXT = "\n\nYou have gathered the following helpful information through previous actions:\n{necessary_context}"
RE_ALFWORLDTASK = re.compile(r"Your task is to:\s*(.*?\.)", re.DOTALL)

def parse_ingredients(text: str) -> str:
    context_trigger = "Gather all following ingredients and follow the directions to prepare this tasty meal."
    ctx = None
    if context_trigger in text:
        s = text.split(context_trigger, 1)[1].replace("\r", "")
        ctx = s.rstrip("\n").lstrip("\n")
        ctx = f"Your recipe is: {ctx}."
    return ctx

def parse_alfworld_task(text: str) -> str:
    m = RE_ALFWORLDTASK.search(text)
    ctx = None
    if m:
        task = m.group(1).strip()
        ctx = f"Your task is: {task}"
    return ctx

def get_necessary_context(necessary_context_list):
    if len(necessary_context_list) == 0:
        return ""
    return NECESSARY_CONTEXT.format(necessary_context="\n\n".join(necessary_context_list.values()))