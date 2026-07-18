import json
import os
from typing import List
from datasets import load_dataset

def VideoMME2qgate_json(video_root: str) -> List[dict]:
    """Load and transform the dataset into the required format for the grounding search.

    The output JSON structure is like:
    [
        {
            "video_path": "path/to/video1.mp4",
            "question": "What is the color of my couch?",
            "options": "A) Red\nB) Black\nC) Green\nD) White\n",
            // More user-defined keys...
        },
        // More entries...
    ]
    """

    dataset = load_dataset("lmms-lab/Video-MME")
    VideoMME_testset = dataset["test"]
    records = []

    for idx, entry in enumerate(VideoMME_testset):
        try:
            # Extract necessary fields from the entry
            video_id = entry.get("videoID")
            question = entry.get("question")
            answer = entry.get("answer", "")
            options_list = entry.get("options", [])
            gt_frame_index = entry.get("frame_indexes", []) # ground-truth frame index for the question            
            duration = entry.get("duration")
            position = entry.get("position", [])

            # Validate required fields
            if not video_id or not question or not options_list:
                raise ValueError(f"Missing required fields in entry {idx+1}. Skipping entry.")

            # Parse the options string into a dictionary
            if options_list:

                # Format the options with letter prefixes (A, B, C, D...)
                options = ""
                for option in options_list:
                    options += option[0] + ') ' + option[3: ] + '\n'

                options = options.rstrip('\n')  # Remove the trailing newline

            # Construct the transformed dictionary for the entry
            transformed_entry = {
                "video_id": video_id,
                "video_path": os.path.join(video_root, f"{video_id}.mp4"),  # Build the full video path
                "question": question,
                "options": options,
                "answer": answer,
                "gt_frame_index": gt_frame_index,
                "duration_group": duration,
                "position": position,
            }

            # Add the transformed entry to the result list
            records.append(transformed_entry)

        except ValueError as e:
            print(f"Skipping entry {idx+1}, reason: {str(e)}")
        except Exception as e:
            print(f"Error processing entry {idx+1}: {str(e)}")

    return records

def LongVideoBench2qgate_json(video_root: str, anno_path: str = "./Datasets/LVBench/lvb_val.json") -> List[dict]:
    """Load and transform the dataset into the required format for the grounding search.

    The output JSON structure is like:
    [
        {
            "video_path": "path/to/video1.mp4",
            "question": "What is the color of my couch?",
            "options": "A) Red\nB) Black\nC) Green\nD) White\n",
            // More user-defined keys...
        },
        // More entries...
    ]
    """

    with open(anno_path, 'r', encoding='utf-8') as file:
        lvb_dataset = json.load(file)
    
    # List to hold the transformed data
    records = []
    num2letter = ['A', 'B', 'C', 'D', 'E']
    # Iterate over each row in the dataset
    for idx, entry in enumerate(lvb_dataset):
        try:
            # Extract necessary fields from the entry
            video_id = entry.get("video_id")
            video_path = entry.get("video_path")
            question = entry.get("question")
            answer = entry.get("correct_choice", "")
            answer = num2letter[answer]
            question_category = entry.get("question_category")
            duration_group = entry.get("duration_group")
            position = entry.get("position", [])
            options_list = entry.get("candidates", "")

            # Validate required fields
            if not video_id or not question or not options_list:
                raise ValueError(f"Missing required fields in entry {idx+1}. Skipping entry.")

            # Parse the options string into a dictionary
            if options_list:
                options = ""

                # Format the options with letter prefixes (A, B, C, D...)
                for i in range(len(options_list)):
                    options += num2letter[i] + ') ' + options_list[i] + '\n'
                
                options = options.rstrip('\n')  # Remove the trailing newline

            # Construct the transformed dictionary for the entry
            transformed_entry = {
                "video_id": video_id,
                "video_path": os.path.join(video_root, video_path),  # Build the full video path
                "question": question,
                "options": options,
                "answer": answer,
                "duration_group": duration_group,
                "position": position,
                "question_category": question_category,
            }

            # Add the transformed entry to the result list
            records.append(transformed_entry)

        except ValueError as e:
            print(f"Skipping entry {idx+1}, reason: {str(e)}")
        except Exception as e:
            print(f"Error processing entry {idx+1}: {str(e)}")

    return records
