import json
from pantheon.package import create_synthetic_package
from pantheon.hashing import sha256_file

def test_package_manifest_matches_data(tmp_path):
    p=create_synthetic_package(tmp_path/'pkg',seed=3,n=50)
    m=json.loads((p/'package_manifest.json').read_text())['files']
    assert m['data.csv']==sha256_file(p/'data.csv')
    assert (p/'claim.yaml').exists()
