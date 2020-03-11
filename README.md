preprocess.py
	When to run: 
		Add binary vector and form final tag vector for each user
	Options:
		--dir: name of the directory containing input data (Default: ‘data’)
		--input_subdir: name of the subdirectory containing input data (Default: ‘raw’)
		--output_subdir: name of the subdirectory for output data (Default: ‘preprocessed’)
		--datetime: datetime to be added to input/output file name (Default: None)	
	Input Files:
		dir/input_subdir/names_tags_raw_datetime.csv
		dir/input_subdir/activities_tags_raw_datetime.csv
		dir/input_subdir/subjects_tags_raw_datetime.csv
		dir/input_subdir/ user_item_raw_datetime.csv
	Output Files:
		dir/output_subdir/names_tags_preprocessed_datetime.csv
		dir/output_subdir/activities_tags_preprocessed_datetime.csv
		dir/output_subdir/subjects_tags_preprocessed_datetime.csv
		dir/output_subdir/ user_item_preprocessed_datetime.csv

RecommendToTopKUser.py
	When to run:
		New Activity to notify topK users
	Options:
		--dir: name of the directory containing input data (Default: ‘data’)
		--input_subdir: name of the subdirectory containing input data (Default: ‘preprocessed’)
		--output_subdir: name of the subdirectory for output data (Default: ‘prediction’)
		--datetime: datetime to be added to input/output file name (Default: None)
		--topK: amount of users to recommended per item(Default: 100)
	Input Files:
		dir/output_subdir/names_tags_preprocessed_datetime.csv
		dir/output_subdir/activities_tags_preprocessed_datetime.csv
	Output Files:
		dir/output_subdir/Recommend_topK_user_datetime.csv

RecommendToTopKItem.py
	When to run:
		Recommend topK activities to new user with less than 3 interactions
	Options:
		--dir: name of the directory containing input data (Default: ‘data’)
		--input_subdir: name of the subdirectory containing input data (Default: ‘preprocessed’)
		--output_subdir: name of the subdirectory for output data (Default: ‘prediction’)
		--datetime: datetime to be added to input/output file name (Default: None)
		--topK: amount of items to recommended per user (Default: 10)
	Input Files:
		dir/input_subdir/names_tags_preprocessed_datetime.csv
		dir/input_subdir/activities_tags_preprocessed_datetime.csv
	Output Files:
		dir/output_subdir/Recommend_topK_item_datetime.csv

NeuMF.py
	When to run:
		When all users in user_item have more than or equal to 3 interactions with an activity
	Options:
		--dir: name of the directory containing input data (Default: ‘data’)
		--input_subdir: name of the subdirectory containing input data (Default: ‘preprocessed’)
		--output_subdir: name of the subdirectory for output data (Default: ‘prediction’)
		--datetime: datetime to be added to input/output file name (Default: None)
		--topK: amount of items to recommended per user (Default: 10)
		--epochs: number of epochs to train the model (Default: 100)
		--batch_size: batch size used to feed to model (Default:256)
	Input:
		dir/input_subdir/names_tags_preprocessed_datetime.csv
		dir/input_subdir/activities_tags_preprocessed_datetime.csv
		dir/input_subdir/subjects_tags_preprocessed_datetime.csv
		dir/input_subdir/user_item_preprocessed_datetime.csv
	Output:
		dir/output_subdir/Model_topK_Recommendation_datetime.csv

