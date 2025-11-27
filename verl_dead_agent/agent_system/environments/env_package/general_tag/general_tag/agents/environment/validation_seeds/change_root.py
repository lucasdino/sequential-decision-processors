from pathlib import Path

files = ["alfworld_valid_seen.txt", "alfworld_valid_unseen.txt"]
old_text = "/home/lucas/"
new_text = "/root/"

for fname in files:
    p = Path(fname)
    txt = p.read_text()
    p.write_text(txt.replace(old_text, new_text))