from torchreid import models, data, optim, engine

datamanager = data.VideoDataManager(
    root='data',
    sources='ilidsvid',
    height=256,
    width=128,
    combineall=False,
    batch_size_train=13,  # number of tracklets
    batch_size_test=31,
    seq_len=19  # number of images in each tracklet
)

train_loader = datamanager.train_loader
test_loader = datamanager.test_loader

# model = models.build_model(
#     name='osnet_x0_25',
#     num_classes=datamanager.num_train_pids,
#     loss='softmax'
# )
# model = model.cuda()
#
# optimizer = optim.build_optimizer(model, optim='adam', lr=0.0003)
#
# scheduler = optim.build_lr_scheduler(optimizer, lr_scheduler='single_step', stepsize=20)
#
# engine = engine.VideoSoftmaxEngine(datamanager, model, optimizer, scheduler=scheduler, pooling_method='avg')

if __name__ == '__main__':

    for data in train_loader:
        print("stop")
        print(data)

    # engine.run(
    #     max_epoch=20,
    #     save_dir='log/resnet50-softmax-mars',
    #     print_freq=20
    # )
