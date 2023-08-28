import lmdb
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import pickle
import os


def write_to_lmdb(db, key, value):
    """
    Write (key,value) to db
    """
    success = False
    while not success:
        txn = db.begin(write=True)
        try:
            txn.put(key, value)
            txn.commit()
            success = True
        except lmdb.MapFullError:
            txn.abort()
            # double the map_size
            curr_limit = db.info()['map_size']
            new_limit = curr_limit*2
            print('>>> Doubling LMDB map size to %sMB ...' % (new_limit>>20,))
            db.set_mapsize(new_limit) # double it

def read_lmdb(filename):
    lmdb_env = lmdb.open(filename)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()

    n_samples=0
    with lmdb_env.begin() as lmdb_txn:
        with lmdb_txn.cursor() as lmdb_cursor:
            for key, value in lmdb_cursor:  
                print(key.decode())
                value = np.asarray(bytearray(value), dtype="uint8")
                value = cv2.imdecode(value, flags=cv2.IMREAD_COLOR)
                cv2.imwrite('value.jpg', value.astype(np.uint8))
                break

def open_pickle(fp):
    import pickle
    fp = os.path.join(os.path.dirname(__file__), fp)
    with open(fp, 'rb') as f:
        data = pickle.load(f)
    return data


class LMDBReader:
    def __init__(self, lmdb_filename, debug=False, mode='visual'):
        super().__init__()
        self.mode = mode
        self.debug = debug
        self.lmdb_filename = lmdb_filename
        print(self.lmdb_filename)
        lmdb_env = lmdb.open(lmdb_filename)
        self.lmdb_txn = lmdb_env.begin()
        self.lmdb_cursor = self.lmdb_txn.cursor()
        
        self.word2id_dict = open_pickle('word2id.pkl')
        self.write_flist()

    def read_key(self, key_name):
        value = self.lmdb_txn.get(str(key_name).encode())
        # value = np.asarray(bytearray(value), dtype="uint8")
        # value = cv2.imdecode(value, flags=cv2.IMREAD_COLOR)
        # cv2.imwrite('{}.jpg'.format(key_name), value.astype(np.uint8))
        if self.mode == 'visual':
            return Image.open(BytesIO(value))
        elif self.mode == 'audio':
            return pickle.loads(value)

    def get_video_flist(self,):
        video_ids = set()
        self.frame_key_list = []
        for i, (key, value) in tqdm(enumerate(self.lmdb_cursor)):  
            frame_key = key.decode()
            self.frame_key_list.append(frame_key)
            word_id = frame_key.split('_')[0]
            video_ids.add(frame_key[:-7])
            if self.debug and i > 100*29: break
        return list(video_ids)
    
    def get_audio_flist(self,):
        video_ids = set()
        for i, (key, value) in tqdm(enumerate(self.lmdb_cursor)):  
            frame_key = key.decode()
            word_id = frame_key.split('_')[0]
            video_ids.add(frame_key)
            if self.debug and i > 100*29: break
        return list(video_ids)

    def write_flist(self, fp=None):
        if self.mode == 'visual':
            video_ids = self.get_video_flist()
        elif self.mode == 'audio':
            video_ids = self.get_audio_flist()
        else:
            raise ValueError
        lines = []
        word_ids = []
        self.items = []
        for i, video_id in tqdm(enumerate(video_ids)):
            word = video_id.split('_')[0]
            word_id = self.word2id_dict[word]
            # word_ids.append(word_id)
            line = [video_id, str(word_id)]
            self.items.append(line)
            writeline = ' '.join(line)
            lines.append(writeline+'\n')
            if self.debug and i > 100*29: break

        if fp is not None:
            with open(fp, 'w') as f:
                f.writelines(lines)

    def __len__(self):
        return len(self.items)
    
    def get_item(self, idx):
        idx = idx % len(self.items)
        return self.items[idx]

    def get_key_name(self, idx):
        idx = idx % len(self.items)
        return self.frame_key_list[idx]
