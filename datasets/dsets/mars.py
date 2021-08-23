import os
from scipy.io import loadmat

from datasets.tools import read_lines, check_files
from datasets.dataset import Dataset


class Mars(Dataset):
    """
    MARS Dataset.

    Reference:
        Zheng et al. MARS: A Video Benchmark for Large-Scale Person Re-identification. ECCV 2016.
    URL: `<http://www.liangzheng.com.cn/Project/project_mars.html>`_

    Dataset statistics:
        - identities: 1261.
        - tracklets: 8298 (train) + 1980 (query) + 9330 (gallery).
        - cameras: 6.
    """

    def __init__(self, dataset_dir='', **kwargs):
        self.dataset_dir = os.path.abspath(os.path.expanduser(dataset_dir))

        self.train_name_path = os.path.join(self.dataset_dir, 'info/train_name.txt')
        self.test_name_path = os.path.join(self.dataset_dir, 'info/test_name.txt')
        self.track_train_info_path = os.path.join(self.dataset_dir, 'info/tracks_train_info.mat')
        self.track_test_info_path = os.path.join(self.dataset_dir, 'info/tracks_test_info.mat')
        self.query_IDX_path = os.path.join(self.dataset_dir, 'info/query_IDX.mat')

        required_files = [
            self.dataset_dir, self.train_name_path, self.test_name_path,
            self.track_train_info_path, self.track_test_info_path,
            self.query_IDX_path
        ]
        check_files(required_files)

        train_names = read_lines(self.train_name_path)
        test_names = read_lines(self.test_name_path)
        track_train = loadmat(self.track_train_info_path)['track_train_info']  # numpy.ndarray (8298, 4)
        track_test = loadmat(self.track_test_info_path)['track_test_info']  # numpy.ndarray (12180, 4)
        query_IDX = loadmat(self.query_IDX_path)['query_IDX'].squeeze()  # numpy.ndarray (1980,)
        query_IDX -= 1  # index from 0
        track_query = track_test[query_IDX, :]
        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
        track_gallery = track_test[gallery_IDX, :]

        train = self.process_data(train_names, track_train, home_dir='bbox_train', relabel=True)
        query = self.process_data(test_names, track_query, home_dir='bbox_test', relabel=False)
        gallery = self.process_data(test_names, track_gallery, home_dir='bbox_test', relabel=False)

        super(Mars, self).__init__(train, query, gallery, **kwargs)

    def process_data(self, names, metadata, home_dir=None, relabel=False, min_seq_len=0):
        assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = metadata.shape[0]
        pid_list = list(set(metadata[:, 2].tolist()))

        tracklets = []
        pid2label = {}

        if relabel:
            pid2label = {pid: label for label, pid in enumerate(pid_list)}

        for tracklet_idx in range(num_tracklets):
            data = metadata[tracklet_idx, ...]
            start_index, end_index, pid, cam_id = data
            if pid == -1:
                continue  # junk images are just ignored
            assert 1 <= cam_id <= 6
            if relabel:
                pid = pid2label[pid]
            cam_id -= 1  # index starts from 0
            img_names = names[start_index - 1:end_index]

            # make sure image names correspond to the same person
            p_names = [img_name[:4] for img_name in img_names]
            assert len(set(p_names)) == 1, 'Error: a single tracklet contains different person images'

            # make sure all images are captured under the same camera
            cam_names = [img_name[5] for img_name in img_names]
            assert len(set(cam_names)) == 1, 'Error: images are captured under different cameras!'

            # append image names with directory information
            img_paths = [os.path.join(self.dataset_dir, home_dir, img_name[:4], img_name) for img_name in img_names]
            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                img_metadata = {'pid': pid, 'cam_id': cam_id}
                tracklets.append((img_paths, img_metadata))

        return tracklets


def test():
    data = Mars('D:/Projects/person-reid/data/mars', track_len=7, track_sampler='random')

    print('Number of cams: {}'.format(data.get_num_cams()))
    print('Number of persons: {}'.format(data.get_num_pids()))
    print('Dataset length: {}'.format(len(data)))
    print('Image shape: {}'.format(data[0][0].shape))


if __name__ == '__main__':
    test()


