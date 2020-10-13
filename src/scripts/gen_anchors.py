'''
Created on Mar 08, 2019

@author: Deepak Ghimire [ghmdeepak@gmail.com]
'''

import os, sys
import argparse
import numpy as np
import random

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

from src.utils.kitti_utils import get_target2

width = 1024
length = 512

def IOU(x, centroids):
    similarities = []
    k = len(centroids)
    for centroid in centroids:
        c_w, c_h = centroid
        w, h = x
        if c_w >= w and c_h >= h:
            similarity = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:
            similarity = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w * h / (w * h + c_w * (c_h - h))
        else:  # means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w * c_h) / (w * h)
        similarities.append(similarity)  # will become (k,) shape
    return np.array(similarities)

def avg_IOU(X, centroids):
    n, d = X.shape
    sum = 0.
    for i in range(X.shape[0]):
        # note IOU() will return array which contains IoU for each centroid and X[i] // slightly ineffective, but I am too lazy
        sum += max(IOU(X[i], centroids))
    return sum / n

def write_anchors_to_file(centroids, X, anchor_file):
    f = open(anchor_file, 'w')

    anchors = centroids.copy()
    print(anchors.shape)

    for i in range(anchors.shape[0]):
        anchors[i][0] *= (width / 32.)
        anchors[i][1] *= (length / 32.)

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    print('Anchors = ', anchors[sorted_indices])

    for i in sorted_indices[:-1]:
        f.write('%0.2f,%0.2f, ' % (anchors[i, 0], anchors[i, 1]))

    # there should not be comma after last anchor, that's why
    f.write('%0.2f,%0.2f\n' % (anchors[sorted_indices[-1:], 0], anchors[sorted_indices[-1:], 1]))

    f.write('%f\n' % (avg_IOU(X, centroids)))
    print()


def kmeans(X, centroids, eps, anchor_file):
    N = X.shape[0]
    iterations = 0
    k, dim = centroids.shape
    prev_assignments = np.ones(N) * (-1)
    iter = 0
    old_D = np.zeros((N, k))

    while True:
        D = []
        iter += 1
        for i in range(N):
            d = 1 - IOU(X[i], centroids)
            D.append(d)
        D = np.array(D)  # D.shape = (N,k)

        print("iter {}: dists = {}".format(iter, np.sum(np.abs(old_D - D))))

        # assign samples to centroids
        assignments = np.argmin(D, axis=1)

        if (assignments == prev_assignments).all():
            print("Centroids = ", centroids)
            write_anchors_to_file(centroids, X, anchor_file)
            return

        # calculate new centroids
        centroid_sums = np.zeros((k, dim), np.float)
        for i in range(N):
            centroid_sums[assignments[i]] += X[i]
        for j in range(k):
            centroids[j] = centroid_sums[j] / (np.sum(assignments == j))

        prev_assignments = assignments.copy()
        old_D = D.copy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', default='../../data/KITTI/ImageSets/train.txt', help='path to filelist\n')
    parser.add_argument('--output_dir', default='anchors', type=str, help='output dir to save anchor file')
    parser.add_argument('--num_clusters', default=3, type=int, help='number of clusters\n')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    f = open(args.filelist)

    lines = [line.rstrip('\n') for line in f.readlines()]
    annotation_dims = []

    for line in lines:

        line = line+".txt"
        label_file = os.path.join('F:/3D-Object-Detection/dataset/KITTI/object/training/label_2', line)

        target = get_target2(label_file)

        nTrueBox = target.shape[0] # 50
        for t in range(nTrueBox):
            if target[t][1] == 0:
                continue

            l = target[t][3]
            w = target[t][4]

            annotation_dims.append(tuple(map(float,(l,w))))
    annotation_dims = np.array(annotation_dims)

    v = np.mean(annotation_dims, axis=0)
    #v[0] *= (width / 32.)
    #v[1] *= (length / 32.)
    #print(v)

    eps = 0.005

    if args.num_clusters == 0:
        for num_clusters in range(1,10):
            anchor_file = os.path.join(args.output_dir, 'anchors_%d.txt' % (num_clusters))
            indices = [ random.randrange(annotation_dims.shape[0]) for i in range(num_clusters)]
            centroids = annotation_dims[indices]
            kmeans(annotation_dims, centroids, eps, anchor_file)
            print('centroids.shape', centroids.shape)
    else:
        anchor_file = os.path.join(args.output_dir, 'anchors_%d.txt' % (args.num_clusters))
        indices = [random.randrange(annotation_dims.shape[0]) for i in range(args.num_clusters)]
        centroids = annotation_dims[indices]
        kmeans(annotation_dims, centroids, eps, anchor_file)
        print('centroids.shape', centroids.shape)

if __name__=="__main__":
    main()

'''
def calc_anchors(l, w):
    w1 = (w/80) * (1024/32)
    l1 = (l/40) * (512/32)
    return l1, w1

print("car: [l,w]", calc_anchors(4.46, 2.07))
print("Ped: [l,w]", calc_anchors(1.33, 2.25))
print("Cycl: [l,w]", calc_anchors(2.27, 2.22))
'''