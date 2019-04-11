#!/usr/bin/env python
import imagehash
import os
import json
import shutil
import multiprocessing
import time
import numpy as np
from PIL import Image


IMAGEDIR = 'images'


def connected_components(neighbors):
    """
    Compute connected components from an adjacency list
    Based on https://stackoverflow.com/a/13837045
    :param neighbors: map vertex -> list of neighbors
    :return: iterator over the connected components (as sets of verteces)
    """
    seen = set()

    def component(node):
        result = set()
        nodes = {node}
        while nodes:
            node = nodes.pop()
            seen.add(node)
            nodes |= neighbors[node] - seen
            result.add(node)
        return result

    for node in neighbors:
        if node not in seen:
            yield component(node)


ALGORITHMS = {
    'phash_8': (lambda i: imagehash.phash(i, hash_size=8), 64),
    'avhash_8': (lambda i: imagehash.average_hash(i, hash_size=8), 64),
    'dhash_8': (lambda i: imagehash.dhash(i, hash_size=8), 256),
    'phash_16': (lambda i: imagehash.phash(i, hash_size=16), 256),
    'avhash_16': (lambda i: imagehash.average_hash(i, hash_size=16), 256),
    'dhash_16': (lambda i: imagehash.dhash(i, hash_size=16), 256),
}

MAX_THRESHOLD = 10

OUTPUT = 'output'


def _hash(arg):
    i, f, algo = arg
    hashfn = ALGORITHMS[algo][0]
    print(i, f)
    i = Image.open(os.path.join(IMAGEDIR, f))
    hash = hashfn(i)
    return f, hash.hash.tolist()


def go(algorithm):
    """
    Compute hashes (and store) them, and then run clustering for a particular hash algorithm
    :param algorithm: the key for the algorithm to run (@see ALGORITHMS)
    """
    start_time = time.time()
    output = os.path.join(OUTPUT, algorithm)
    print('Running for', algorithm, ', writing to ', output)

    hashfn, max_threshold = ALGORITHMS[algorithm]

    hashes = {}

    hashes_json_path = os.path.join(output, 'hashes.json')
    if os.path.exists(hashes_json_path):
        print('Loading from ', hashes_json_path)
        with open(hashes_json_path) as inf:
            data = json.load(inf)
        for f, h in data.items():
            hashes[f] = imagehash.ImageHash(np.array(h))
    else:
        pool = multiprocessing.Pool(processes=max(1, os.cpu_count() - 1))
        l = [(i, f, algorithm) for i, f in enumerate(sorted(os.listdir(IMAGEDIR)))]
        result = pool.imap_unordered(_hash, l)
        # result = map(_hash, enumerate(sorted(os.listdir(IMAGEDIR))))
        hashes = {f: imagehash.ImageHash(np.array(h)) for f, h in result}

    files = set(hashes.keys())

    hashes_sorted = sorted(hashes.items(), key=lambda h: str(h[1]))
    diffs = []
    for i1, h1 in enumerate(hashes_sorted):
        for i2, h2 in enumerate(hashes_sorted):
            if i1 < i2:
                diffs.append((h1[1] - h2[1], h1[0], h2[0]))
    diffs = sorted(diffs, key=lambda d: d[0])

    os.makedirs(output, exist_ok=True)
    hashes_to_write = {}
    for f, h in hashes_sorted:
        hashes_to_write[f] = h.hash.tolist()
    with open(hashes_json_path, 'w') as outf:
        json.dump(hashes_to_write, outf)

    with open(os.path.join(output, 'diffs.txt'), 'w') as outf:
        for d in diffs:
            print(d, file=outf)

    print('Min diffs:\n', '\n'.join(str(d) for d in diffs[:20]))
    print('Max diffs:\n', '\n'.join(str(d) for d in diffs[-20:]))
    print('Avg diff: ', sum(d[0] for d in diffs) / len(diffs))

    for threshold in range(0, max_threshold):
        print('Copying for threshold {}'.format(threshold))

        neighbors = {}
        for d in diffs:
            if d[0] <= threshold:
                neighbors.setdefault(d[1], set()).add(d[2])
                neighbors.setdefault(d[2], set()).add(d[1])

        clusters = list(sorted(
            connected_components(neighbors),
            key=lambda c: len(c),
            reverse=True))

        print(clusters)
        in_clusters = set().union(*clusters)
        unclustered = files - in_clusters

        destdir = os.path.join(output, 'thr_{}_unclustered_{}'.format(
            str(threshold).zfill(3), len(unclustered)))
        shutil.rmtree(destdir, ignore_errors=True)
        os.makedirs(destdir)

        for cnt, cluster in enumerate(clusters + [unclustered]):
            if cnt == len(clusters):
                name = 'unclustered'
            else:
                name = str(cnt + 1).zfill(3)
            cdir = os.path.join(destdir, '{}_{}'.format(name, len(cluster)))
            os.makedirs(cdir)
            for f in cluster:
                fname, ext = os.path.splitext(f)
                filename = '{}_{}{}'.format(fname, str(hashes[f]), ext)
                os.symlink(os.path.abspath(os.path.join(IMAGEDIR, f)), os.path.join(cdir, filename))
        
        if not unclustered:
            # no need to raise threshold higher
            break

    end_time = time.time()

    print('Time taken: ', round(end_time - start_time, 2), 'sec\n\n')


for algorithm in ALGORITHMS.keys():
    go(algorithm)
