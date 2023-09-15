import pandas as pd
import tqdm
import random
import os
#%%
movie_data = pd.read_json('../dataset/reviews_Movies_and_TV_5.json', lines=True)
movie_meta = pd.read_json('../dataset/meta_Movies_and_TV.json', lines=True)

merged_movie = pd.merge(movie_data, movie_meta, on='asin', how='left')

columns_to_keep = ['reviewerID', 'asin', 'overall', 'category', 'description']
movie = merged_movie[columns_to_keep]
movie = movie.rename(columns={'reviewerID': 'uid', 'asin': 'iid', 'overall': 'y'})
#%%
electronics_data = pd.read_json('../dataset/reviews_Electronics_5.json', lines=True)
electronics_meta = pd.read_json('../dataset/meta_Electronics.json', lines=True)

merged_electronics = pd.merge(electronics_data, electronics_meta, on='asin', how='left')

columns_to_keep = ['reviewerID', 'asin', 'overall', 'category', 'description']
electronics = merged_electronics[columns_to_keep]
electronics = electronics.rename(columns={'reviewerID': 'uid', 'asin': 'iid', 'overall': 'y'})
#%%
def mapper(src, tgt):
        print('Source inters: {}, uid: {}, iid: {}.'.format(len(src), len(set(src.uid)), len(set(src.iid))))
        print('Target inters: {}, uid: {}, iid: {}.'.format(len(tgt), len(set(tgt.uid)), len(set(tgt.iid))))
        co_uid = set(src.uid) & set(tgt.uid)
        all_uid = set(src.uid) | set(tgt.uid)
        print('All uid: {}, Co uid: {}.'.format(len(all_uid), len(co_uid)))
        uid_dict = dict(zip(all_uid, range(len(all_uid))))
        iid_dict_src = dict(zip(set(src.iid), range(len(set(src.iid)))))
        iid_dict_tgt = dict(zip(set(tgt.iid), range(len(set(src.iid)), len(set(src.iid)) + len(set(tgt.iid)))))
        src.uid = src.uid.map(uid_dict)
        src.iid = src.iid.map(iid_dict_src)
        tgt.uid = tgt.uid.map(uid_dict)
        tgt.iid = tgt.iid.map(iid_dict_tgt)
        return src, tgt
    
def get_history(data, uid_set):
        pos_seq_dict = {}
        for uid in tqdm.tqdm(uid_set):
            pos = data[(data.uid == uid) & (data.y > 3)].iid.values.tolist()
            pos_seq_dict[uid] = pos
        return pos_seq_dict

def split(src, tgt, ratio):
    print('All iid: {}.'.format(len(set(src.iid) | set(tgt.iid))))
    src_users = set(src.uid.unique())
    tgt_users = set(tgt.uid.unique())
    co_users = src_users & tgt_users
    test_users = set(random.sample(co_users, round(ratio[1] * len(co_users))))
    train_src = src
    train_tgt = tgt[tgt['uid'].isin(tgt_users - test_users)]
    test = tgt[tgt['uid'].isin(test_users)]
    pos_seq_dict = get_history(src, co_users)
    train_meta = tgt[tgt['uid'].isin(co_users - test_users)]
    train_meta['pos_seq'] = train_meta['uid'].map(pos_seq_dict)
    test['pos_seq'] = test['uid'].map(pos_seq_dict)
    return train_src, train_tgt, train_meta, test

def save(train_src, train_tgt, train_meta, test, ratio, src, tgt):
        output_root = '../project_data/' + str(int(ratio[0] * 10)) + '_' + str(int(ratio[1] * 10)) + \
                      '/tgt_' + tgt + '_src_' + src
        if not os.path.exists(output_root):
            os.makedirs(output_root)
        print(output_root)
        train_src.to_csv(output_root + '/train_src.csv', sep=',', header=None, index=False)
        train_tgt.to_csv(output_root + '/train_tgt.csv', sep=',', header=None, index=False)
        train_meta.to_csv(output_root +  '/train_meta.csv', sep=',', header=None, index=False)
        test.to_csv(output_root + '/test.csv', sep=',', header=None, index=False)
#%%
ratio_information = [[0.8, 0.2], [0.5,0.5], [0.3, 0.7]]

for ratio in ratio_information:

    src = movie
    tgt = electronics
    src, tgt = mapper(src, tgt)
    train_src, train_tgt, train_meta, test = split(src, tgt, ratio)
    save(train_src, train_tgt, train_meta, test, ratio, "movie", "electronics")












