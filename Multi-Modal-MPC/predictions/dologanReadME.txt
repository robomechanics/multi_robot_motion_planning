Instructions to generate predictions:

Instructions to Run:
python pred_utils.py --eval_data_dict=<eval_data>.pkl --iter_num=<epochs> --log_dir=../experiments/pedestrians/models/ --trained_model_dir=<trained_model_name> --eval_task=load_dataset

Notes:
I made a copy of the Uncontrolled Agent File and placed it into the predictions folder since I couldn't figure out how the relative path stuff.

I defined an instance of Uncontrolled Agent in load_dataset, but you might want to do that elsewhere in your code. 

To animate, uncomment line 318 of pred_utils.

