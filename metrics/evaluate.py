import os
import torch
import numpy as np
from torch.nn import functional as F

from metrics.rank import evaluate_rank
from metrics.distance import compute_distance
from metrics.visualize import visualize_ranks


@torch.no_grad()
def eval_feature_extractor(feature_extractor, data_loader, use_gpu):
    f_, pids_, cam_ids_ = [], [], []
    for batch_idx, data in enumerate(data_loader):
        imgs = data[0]
        pids = data[1]['pid']
        cam_ids = data[1]['cam_id']
        if use_gpu:
            imgs = imgs.cuda()
        features = feature_extractor.extract_from_tracklets(imgs)
        f_.append(features)
        pids_.extend(pids)
        cam_ids_.extend(cam_ids)
    f_ = torch.cat(f_, 0)
    pids_ = np.asarray(pids_)
    cam_ids_ = np.asarray(cam_ids_)
    return f_, pids_, cam_ids_


@torch.no_grad()
def evaluate(
        feature_extractor,
        datamanager,
        dist_metric='euclidean',
        normalize=False,
        visrank=False,
        visrank_topk=10,
        single_shot_metric=False,
        ranks=(1, 5, 10),
        use_gpu=True
):
    """
    Pipeline for evaluating a model. Computes CMC curves and mAP.

    Args:
        feature_extractor (reid.ReIDFeatureExtractor): reid feature extractor.
        datamanager (datasets.DataManager): data manager.
        dist_metric (str, optional): distance metric used to compute distance matrix
            between query and gallery. Default is "euclidean".
        normalize (bool, optional): performs L2 normalization on feature vectors before
            computing feature distance. Default is False.
        visrank (bool, optional): visualizes ranked results. Default is False. It is recommended to
            enable ``visrank`` when ``test_only`` is True. The ranked images will be saved to
            "save_dir/visrank_dataset", e.g. "save_dir/visrank_market1501".
        visrank_topk (int, optional): top-k ranked images to be visualized. Default is 10.
        single_shot_metric (bool, optional): use single-gallery-shot setting for cuhk03.
            Default is False. This should be enabled when using cuhk03 classic split.
        ranks (list, optional): cmc ranks to be computed. Default is [1, 5, 10, 20].
            Default is False. This is only enabled when test_only=True.
        use_gpu (bool, optional): default is True.
    """
    query_loader = datamanager.test_loader['query']
    gallery_loader = datamanager.test_loader['gallery']

    # Extracting features from the query set
    qf, q_pids, q_cam_ids = eval_feature_extractor(feature_extractor, query_loader, use_gpu)

    # Extracting features from the gallery set
    gf, g_pids, g_cam_ids = eval_feature_extractor(feature_extractor, gallery_loader, use_gpu)

    if normalize:
        qf = F.normalize(qf, p=2, dim=1)
        gf = F.normalize(gf, p=2, dim=1)

    # compute query-to-gallery distance matrix
    distmat = compute_distance(qf, gf, dist_metric)
    if use_gpu:
        distmat = distmat.cpu()
    distmat = distmat.numpy()

    # Compute CMC and mAP
    cmc, mAP = evaluate_rank(
        distmat,
        q_pids,
        g_pids,
        q_cam_ids,
        g_cam_ids,
        single_shot_metric=single_shot_metric
    )

    print('mAP: {:.1%}'.format(mAP))
    print('CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))

    if visrank:
        visualize_ranks(
            distmat,
            (datamanager.test_set['query'], datamanager.test_set['query']),
            save_dir=os.path.join('ranks_' + datamanager.source),
            top_k=visrank_topk
        )

    return mAP, cmc





