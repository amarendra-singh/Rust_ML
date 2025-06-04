import urllib.request
import gzip
import os
import shutil

urls = [
    ('train-images-idx3-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'),
    ('train-labels-idx1-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'),
    ('t10k-images-idx3-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'),
    ('t10k-labels-idx1-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'),
]

os.makedirs('data', exist_ok=True)

for fname, url in urls:
    out_path = os.path.join('data', fname)
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, out_path)
    print(f"Extracting {out_path} ...")
    with gzip.open(out_path, 'rb') as f_in:
        with open(out_path.replace('.gz', ''), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(out_path)

print("All MNIST files downloaded and extracted to ./data")