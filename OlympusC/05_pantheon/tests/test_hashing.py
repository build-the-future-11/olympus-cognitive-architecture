from pathlib import Path
from pantheon.hashing import sha256_file

def test_sha256_stable(tmp_path):
    p=tmp_path/'x.txt'; p.write_text('abc')
    assert sha256_file(p) == 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
