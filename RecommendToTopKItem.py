import numpy as np
from time import time
import pandas as pd
import time
import ast
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='RecommendToTopKItem')
    parser.add_argument('--dir', type=str, default='data', help='name of the directory containing input data')
    parser.add_argument('--input_subdir', type=str, default='preprocessed', help='name of the subdirectory containing input data')
    parser.add_argument('--output_subdir', type=str, default='prediction', help='name of the subdirectory for output data')
    parser.add_argument('--topK', type=int, default=10, help='amount of items to recommended per user')
    parser.add_argument('--datetime', type=str, default='', help='datetime to be added to input/output file name')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    dir = args.dir
    input_subdir = args.input_subdir
    output_subdir = args.output_subdir
    datetime = args.datetime
    topK = args.topK

    df_names_tags = pd.read_csv(os.path.join(dir,input_subdir,"names_tags_preprocessed_" + datetime + ".csv"), delimiter=',')
    df_activities_tags = pd.read_csv(os.path.join(dir,input_subdir,"activities_tags_preprocessed_" + datetime + ".csv"), delimiter=',')

    str2list_list = [df_names_tags.final_tags,df_activities_tags.TagsVec]

    for change_list in str2list_list:
        for idx,tag in enumerate(change_list):
            change_list[idx] = ast.literal_eval(tag)

    students = []
    classes = []
    activitiesID = []
    activitiesTags = []
    scores = []
    Nos = []

    for idx,student in df_names_tags.iterrows():
        for idx2,activity in df_activities_tags.iterrows():
            score = np.dot(student['final_tags'],activity['TagsVec'])
            count = np.count_nonzero(student['final_tags'])
            Nos.append(student['No'])
            students.append(student['Interests'])
            classes.append(student['Current_Subjects'])
            activitiesID.append(activity['ActivityID'])
            activitiesTags.append(activity['Tags'])
            scores.append(score/count)

    df_result = pd.DataFrame()
    df_result['UserID'] = Nos
    df_result['ActivityID'] = activitiesID
    #df_result['UserInterest'] = students
    #df_result['Classes'] = classes
    #df_result['ActivityTags'] = activitiesTags
    df_result['Score'] = scores

    topKpredictions = df_result.groupby('UserID')['ActivityID','Score'].apply(lambda x: x.nlargest(topK,columns=['Score'])).reset_index()
    topKpredictions.drop(columns=[topKpredictions.columns[1]],inplace=True)

    topKpredictions.to_csv(os.path.join(dir,output_subdir,"Recommend_topK_item_" + datetime + ".csv"),index=False)
