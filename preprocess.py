import argparse
import ast
import os.path
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

def parse_args():
    parser = argparse.ArgumentParser(description='Dots')
    parser.add_argument('--dir', type=str, default='data', help='name of the directory containing input data')
    parser.add_argument('--input_subdir', type=str, default='raw', help='name of the subdirectory containing input data')
    parser.add_argument('--output_subdir', type=str, default='preprocessed', help='name of the subdirectory for output data')
    parser.add_argument('--datetime', type=str, default='', help='datetime to be added to input/output file name')
    return parser.parse_args()

def create_vec(list_input, mlb):

    vec = pd.DataFrame(mlb.fit_transform(list_input),
                       columns=mlb.classes_)

    vec_list = vec.values.tolist()

    return vec_list,mlb

if __name__ == '__main__':
    args = parse_args()
    dir = args.dir
    input_subdir = args.input_subdir
    output_subdir = args.output_subdir
    datetime = args.datetime

    df_names_tags = pd.read_csv(os.path.join(dir,input_subdir,"names_tags_raw_" + datetime + ".csv"), delimiter=',')
    df_activities_tags = pd.read_csv(os.path.join(dir,input_subdir,"activities_tags_raw_" + datetime + ".csv"), delimiter=',')
    df_subjects_tags = pd.read_csv(os.path.join(dir,input_subdir,"subjects_tags_raw_" + datetime + ".csv"), delimiter=',')
    df_user_item = pd.read_csv(os.path.join(dir,input_subdir,"user_item_raw_" + datetime + ".csv"), delimiter=',')

    # Change string list to list
    str2list_list = [df_names_tags.Interests,df_names_tags.Current_Subjects
                    ,df_subjects_tags.tags,df_activities_tags.Tags]

    for change_list in str2list_list:
        for idx,tag in enumerate(change_list):
            change_list[idx] = ast.literal_eval(tag)

    mlb = MultiLabelBinarizer()

    to_vec_list = [df_activities_tags.Tags, df_names_tags.Interests, df_subjects_tags.tags]

    vec_list,mlb = create_vec(list(df_activities_tags.Tags), mlb)
    df_activities_tags['TagsVec'] = vec_list

    vec_list,mlb = create_vec(list(df_names_tags.Interests), mlb)
    df_names_tags['InterestsVec'] = vec_list

    vec_list,mlb = create_vec(list(df_subjects_tags.tags), mlb)
    df_subjects_tags['TagsVec'] = vec_list

    #Generate tags based on current subjects
    additional_tags = []
    for subjects in df_names_tags.Current_Subjects:
        subjects_tags = []
        for subject in subjects:
            [tags] = df_subjects_tags[df_subjects_tags.subjectID == subject].TagsVec.tolist()
            subjects_tags.append(tags)
        zipped = zip(*subjects_tags)
        ans = [sum(i) for i in zipped]
        additional_tags.append(ans)
    df_names_tags['AdditionalVec'] = additional_tags

    #Generate final tags based on interestvec and additionalvec
    final_tags = []
    for idx,row in df_names_tags.iterrows():
        subjects_tags = []
        dot = np.add(2 * np.array(row['InterestsVec']), np.array(row['AdditionalVec'])).tolist()
        final_tags.append(dot)
    df_names_tags['final_tags'] = final_tags


    df_activities_tags.to_csv(os.path.join(dir,output_subdir,"activities_tags_preprocessed_" + datetime + ".csv"), index=False)
    df_names_tags.to_csv(os.path.join(dir,output_subdir,"names_tags_preprocessed_" + datetime + ".csv"), index=False)
    df_subjects_tags.to_csv(os.path.join(dir,output_subdir,"subjects_tags_preprocessed_" + datetime + ".csv"), index=False)
    df_user_item.to_csv(os.path.join(dir,output_subdir,"user_item_preprocessed_" + datetime + ".csv"), index=False)
