from yacs.config import CfgNode as CN


def get_default_config():
    cfg = CN()

    # model
    cfg.model = CN()
    cfg.model.name = 'resnet50'
    cfg.model.pretrained = True # automatically load pretrained model weights if available
    cfg.model.load_weights = '' # path to model weights
    cfg.model.resume = '' # path to checkpoint for resume training
    cfg.model.use_gpu = True

    # data
    cfg.data = CN()
    cfg.data.type = 'image'
    cfg.data.root = 'reid-data'
    cfg.data.sources = ['market1501']
    cfg.data.targets = ['market1501']
    cfg.data.workers = 4 # number of data loading workers
    cfg.data.split_id = 0 # split index
    cfg.data.height = 256 # image height
    cfg.data.width = 128 # image width
    cfg.data.combineall = False # combine train, query and gallery for training
    cfg.data.transforms = ['random_flip'] # data augmentation
    cfg.data.norm_mean = [0.485, 0.456, 0.406] # default is imagenet mean
    cfg.data.norm_std = [0.229, 0.224, 0.225] # default is imagenet std
    cfg.data.save_dir = 'log' # path to save log
    cfg.data.load_train_targets = False

    # specific datasets
    cfg.market1501 = CN()
    cfg.market1501.use_500k_distractors = False # add 500k distractors to the gallery set for market1501
    cfg.cuhk03 = CN()
    cfg.cuhk03.labeled_images = False # use labeled images, if False, use detected images
    cfg.cuhk03.classic_split = False # use classic split by Li et al. CVPR14
    cfg.cuhk03.use_metric_cuhk03 = False # use cuhk03's metric for evaluation

    # sampler
    cfg.sampler = CN()
    cfg.sampler.train_sampler = 'RandomSampler'
    cfg.sampler.num_instances = 4 # number of instances per identity for RandomIdentitySampler

    # video reid setting
    cfg.video = CN()
    cfg.video.seq_len = 15 # number of images to sample in a tracklet
    cfg.video.sample_method = 'evenly' # how to sample images from a tracklet
    cfg.video.pooling_method = 'avg' # how to pool features over a tracklet

    # train
    cfg.train = CN()
    cfg.train.optim = 'adam'
    cfg.train.lr = 0.0003
    cfg.train.weight_decay = 5e-4
    cfg.train.max_epoch = 60
    cfg.train.start_epoch = 0
    cfg.train.batch_size = 32
    cfg.train.fixbase_epoch = 0 # number of epochs to fix base layers
    cfg.train.open_layers = [
        'classifier'
    ] # layers for training while keeping others frozen
    cfg.train.staged_lr = False # set different lr to different layers
    cfg.train.new_layers = ['classifier'] # newly added layers with default lr
    cfg.train.base_lr_mult = 0.1 # learning rate multiplier for base layers
    cfg.train.lr_scheduler = 'single_step'
    cfg.train.stepsize = [20] # stepsize to decay learning rate
    cfg.train.gamma = 0.1 # learning rate decay multiplier
    cfg.train.print_freq = 20 # print frequency
    cfg.train.seed = 1 # random seed

    # optimizer
    cfg.sgd = CN()
    cfg.sgd.momentum = 0.9 # momentum factor for sgd and rmsprop
    cfg.sgd.dampening = 0. # dampening for momentum
    cfg.sgd.nesterov = False # Nesterov momentum
    cfg.rmsprop = CN()
    cfg.rmsprop.alpha = 0.99 # smoothing constant
    cfg.adam = CN()
    cfg.adam.beta1 = 0.9 # exponential decay rate for first moment
    cfg.adam.beta2 = 0.999 # exponential decay rate for second moment

    # loss
    cfg.loss = CN()
    cfg.loss.name = 'softmax'
    cfg.loss.softmax = CN()
    cfg.loss.softmax.label_smooth = True # use label smoothing regularizer
    cfg.loss.triplet = CN()
    cfg.loss.triplet.margin = 0.3 # distance margin
    cfg.loss.triplet.weight_t = 1. # weight to balance hard triplet loss
    cfg.loss.triplet.weight_x = 0. # weight to balance cross entropy loss

    # test
    cfg.test = CN()
    cfg.test.batch_size = 100
    cfg.test.dist_metric = 'euclidean' # distance metric, ['euclidean', 'cosine']
    cfg.test.normalize_feature = False # normalize feature vectors before computing distance
    cfg.test.ranks = [1, 5, 10, 20] # cmc ranks
    cfg.test.evaluate = False # test only
    cfg.test.eval_freq = -1 # evaluation frequency (-1 means to only test after training)
    cfg.test.start_eval = 0 # start to evaluate after a specific epoch
    cfg.test.rerank = False # use person re-ranking
    cfg.test.visrank = False # visualize ranked results (only available when cfg.test.evaluate=True)
    cfg.test.visrank_topk = 10 # top-k ranks to visualize
    cfg.test.visactmap = False # visualize CNN activation maps

    return cfg.clone()
