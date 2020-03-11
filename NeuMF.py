import sys
import time
import os
import ast
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from Dataset import Dataset
import time
from ncf_model import NCF
from python_splitters import python_stratified_split
from tensorflow.keras import backend as K

def parse_args():
    parser = argparse.ArgumentParser(description='NeuMF')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size used to feed to model')
    parser.add_argument('--dir', type=str, default='data', help='name of the directory containing input data')
    parser.add_argument('--input_subdir', type=str, default='preprocessed', help='name of the subdirectory containing input data')
    parser.add_argument('--output_subdir', type=str, default='prediction', help='name of the subdirectory for output data')
    parser.add_argument('--topK', type=int, default=10, help='amount of items to recommended per user')
    parser.add_argument('--datetime', type=str, default='', help='datetime to be added to input/output file name')
    return parser.parse_args()

def topKevaluate(data,model,k=10):
    ndcgs = []
    hit_ratio = []
    ranks = []

    for b in data.test_loader():
        user_input, item_input, labels = b
        output = model.predict(user_input, item_input, is_list=True)

        output = np.squeeze(output)
        rank = sum(output >= output[0])
        if rank <= k:
            ndcgs.append(1 / np.log(rank + 1))
            hit_ratio.append(1)
        else:
            ndcgs.append(0)
            hit_ratio.append(0)
        if rank == 101:
            ranks.append(sum(output > output[0]))
        else:
            ranks.append(rank)

    eval_ndcg = np.mean(ndcgs)
    eval_hr = np.mean(hit_ratio)

    return eval_hr, eval_ndcg

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


if __name__ == '__main__':
    args = parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    dir = args.dir
    input_subdir = args.input_subdir
    output_subdir = args.output_subdir
    datetime = args.datetime
    topK = args.topK

    df_data = pd.read_csv(os.path.join(dir,input_subdir,"user_item_preprocessed_" + datetime + ".csv"), delimiter=',')
    df_names_tags = pd.read_csv(os.path.join(dir,input_subdir,"names_tags_preprocessed_" + datetime + ".csv"), delimiter=',')
    df_activities_tags = pd.read_csv(os.path.join(dir,input_subdir,"activities_tags_preprocessed_" + datetime + ".csv"), delimiter=',')
    df_subjects_tags = pd.read_csv(os.path.join(dir,input_subdir,"subjects_tags_preprocessed_" + datetime + ".csv"), delimiter=',')

    # Change string list to list
    str2list_list = [df_names_tags.InterestsVec,df_names_tags.Interests,df_names_tags.Current_Subjects
                    ,df_names_tags.AdditionalVec,df_names_tags.final_tags,df_subjects_tags.tags
                    ,df_subjects_tags.TagsVec,df_activities_tags.Tags,df_activities_tags.TagsVec]

    for change_list in str2list_list:
        for idx,tag in enumerate(change_list):
            change_list[idx] = ast.literal_eval(tag)

    n_users = df_data.userID.nunique()
    n_items = df_data.itemID.nunique()

    #Splitting data
    train, test = python_stratified_split(df_data, 0.75)

    data = Dataset(train=train, test=test, n_neg_test=50, seed=DEFAULT_SEED, binary=True)

    # Build model
    model = NCF (
    n_users=data.n_users,
    n_items=data.n_items,
    model_type="NeuMF",
    n_factors=4,
    layer_sizes=[32,16,8,4],
    n_epochs=epochs,
    batch_size=batch_size,
    learning_rate=1e-3,
    verbose=1,
    seed=DEFAULT_SEED
    )

    start_time = time.time()

    print("Start training")

    model.fit(data)

    train_time = time.time() - start_time

    print("Took {} seconds for training.".format(train_time))

    # Init performance
    (hr, ndcg) = topKevaluate(data, model, topK)

    print('Init: HR = %.4f, NDCG = %.4f' % (hr, ndcg))

    start_time = time.time()

    users, items, preds = [], [], []
    item = list(train.itemID.unique())
    for user in train.userID.unique():
        user = [user] * len(item)
        users.extend(user)
        items.extend(item)
        preds.extend(list(model.predict(user, item, is_list = True)))

    all_predictions = pd.DataFrame(data={"userID": users, "itemID":items, "prediction":preds})

    merged = pd.merge(train, all_predictions, on=["userID", "itemID"], how="outer")
    all_predictions = merged[merged.rating.isnull()].drop(['rating'], axis=1)

    test_time = time.time() - start_time
    print("Took {} seconds for prediction.".format(test_time))

    idx2item = dict(zip(all_predictions.index, all_predictions.itemID))

    top10predictions = all_predictions.groupby('userID')['itemID','prediction'].apply(lambda x: x.nlargest(10,'prediction'))

    user_name = []
    group_id = []
    group_des = []
    for user, item in top10predictions.index:
        user_name.append(df_names_tags[df_names_tags.index == user].Interests.values[0])
        group_des.append(df_activities_tags[df_activities_tags.index == idx2item[item]].Tags.values[0])

    final_top10_predictions = top10predictions.reset_index()

    final_top10_predictions['UserInterest'] = user_name

    final_top10_predictions['ActivityTags'] = group_des
    final_top10_predictions.drop(columns='level_1', inplace=True)

    final_top10_predictions.to_csv(os.path.join(dir,output_subdir,"Model_topK_Recommendation_" + datetime + ".csv"),index=False)
