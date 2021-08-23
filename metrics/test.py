from reid.feature_extractor import ReidFeatureExtractor
from metrics.model_complexity import compute_complexity
from datasets.datamanager import DataManager
from metrics.evaluate import evaluate


def test():
    reid_extractor = ReidFeatureExtractor(
        model_name='osnet_x1_0',
        cfg_filename='../config/configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml',
    )
    reid_extractor.build_extractor()

    # number of parameters and floating-point operations
    params, flops = compute_complexity(reid_extractor.model, (1, 3, 256, 128), verbose=False)

    data = DataManager(
        root='D:/Projects/person-reid/data',
        source='ilidsvid',
        track_len=20,
        batch_size_test=8
    )

    evaluate(  
        reid_extractor,
        data,
        dist_metric='euclidean',
        normalize=False,
        visrank=False,
        visrank_topk=10,
        single_shot_metric=False,
        ranks=(1, 5, 10),
        use_gpu=True
    )


if __name__ == '__main__':
    test()
