import numpy as np
import shutil
import os

from datasets.tools import mkdir_if_missing


def visualize_ranks(distmat, dataset, save_dir, top_k=10):
    """
    Visualizes ranked results. Ranks will be saved in folders each containing a tracklet.

    Args:
        distmat (np.ndarray): distance matrix of shape (num_query, num_gallery).
        dataset (tuple): a 2-tuple containing (query, gallery), each of which contains
            tuples of (img_paths, metadata).
        save_dir (str): directory to save output images.
        top_k (int, optional): denoting top-k images in the rank list to be visualized.
            Default is 10.
    """
    num_q, num_g = distmat.shape
    mkdir_if_missing(save_dir)

    query, gallery = dataset
    assert num_q == len(query)
    assert num_g == len(gallery)

    indices = np.argsort(distmat, axis=1)

    def copy_img_to(src, dst, rank, prefix, matched_=False):
        """
        Args:
            src: image path or tuple 
            dst: target directory
            rank: int, denoting ranked position, starting from 1
            prefix: string
            matched_: bool
        """
        if isinstance(src, (tuple, list)):
            if prefix == 'gallery':
                suffix = 'TRUE' if matched_ else 'FALSE'
                dst = os.path.join(dst, prefix + '_top' + str(rank).zfill(3)) + '_' + suffix
            else:
                dst = os.path.join(dst, prefix + '_top' + str(rank).zfill(3))
            mkdir_if_missing(dst)
            for img_path in src:
                shutil.copy(img_path, dst)
        else:
            dst = os.path.join(dst, prefix + '_top' + str(rank).zfill(3) + '_name_' + os.path.basename(src))
            shutil.copy(src, dst)

    for q_idx in range(num_q):
        qimg_path, q_mdata = query[q_idx]
        qpid = q_mdata['pid']
        qcamid = q_mdata['cam_id']
        qimg_path_name = qimg_path[0] if isinstance(qimg_path, (tuple, list)) else qimg_path

        qdir = os.path.join(save_dir, os.path.basename(os.path.splitext(qimg_path_name)[0]))
        mkdir_if_missing(qdir)
        copy_img_to(qimg_path, qdir, rank=0, prefix='query')

        rank_idx = 1
        for g_idx in indices[q_idx, :]:
            gimg_path, g_mdata = gallery[g_idx]
            gpid = g_mdata['pid']
            gcamid = g_mdata['cam_id']
            invalid = (qpid == gpid) & (qcamid == gcamid)

            if not invalid:
                matched = gpid == qpid
                copy_img_to(gimg_path, qdir, rank=rank_idx, prefix='gallery', matched_=matched)

                rank_idx += 1
                if rank_idx > top_k:
                    break

