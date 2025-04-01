import os
import torch
import shutil
import wandb

def log_eval_scores(accelerator, args, global_step):
    if accelerator.is_main_process:
        is_txt_path = os.path.join(args.save_dir, "im256_is.txt")
        fid_txt_path = os.path.join(args.save_dir, "im256_fid.txt")
        clip_txt_path = os.path.join(args.save_dir, "im256_clip.txt")

        score_dict = {}

        # Read Inception Score (IS)
        try:
            with open(is_txt_path, "r") as f:
                lines = f.readlines()
                is_score = float(lines[-2].strip().split()[-1])
                score_dict["IS"] = is_score
        except FileNotFoundError:
            print(f"Warning: IS score file {is_txt_path} not found.")
            score_dict["IS"] = None

        # Read Fréchet Inception Distance (FID)
        try:
            with open(fid_txt_path, "r") as f:
                lines = f.readlines()
                fid_score = float(lines[0].strip().split()[-1])
                score_dict["FID"] = fid_score
        except FileNotFoundError:
            print(f"Warning: FID score file {fid_txt_path} not found.")
            score_dict["FID"] = None

        # Read CLIP Score
        try:
            with open(clip_txt_path, "r") as f:
                lines = f.readlines()
                clip_score = float(lines[0].strip().split()[-1])
                score_dict["CLIP"] = clip_score
        except FileNotFoundError:
            print(f"Warning: CLIP score file {clip_txt_path} not found.")
            score_dict["CLIP"] = None

        # Logging scores to WandB
        wandb.log(score_dict, step=global_step)

        # Print scores for verification
        print(f"Inception Score (IS): {score_dict['IS']}")
        print(f"Fréchet Inception Distance (FID): {score_dict['FID']}")
        print(f"CLIP Score: {score_dict['CLIP']}")

        try:
            shutil.rmtree(args.save_dir)
            print(f"All folders in {args.save_dir} have been deleted.")
        except Exception as e:
            print(f"Error occurred while deleting folders in {args.save_dir}: {e}")
    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()
    
    
def log_eval_scores_unseen_setting(accelerator, args, global_step):
    if accelerator.is_main_process:
        # Prepare paths for IS, FID, CLIP scores for each dataset type
        datasets = ['all', 'seen', 'unseen']
        scores = {
            "IS": {},
            "FID": {},
            "CLIP": {}
        }
        
        for dataset_type in datasets:
            is_txt_path = os.path.join(args.save_dir, dataset_type, "im256_is.txt")
            fid_txt_path = os.path.join(args.save_dir, dataset_type, "im256_fid.txt")
            clip_txt_path = os.path.join(args.save_dir, dataset_type, "im256_clip.txt")

            # Read Inception Score (IS)
            try:
                with open(is_txt_path, "r") as f:
                    lines = f.readlines()
                    is_score = float(lines[-2].strip().split()[-1])
                    scores["IS"][dataset_type] = is_score
            except FileNotFoundError:
                print(f"Warning: IS score file {is_txt_path} not found.")
                scores["IS"][dataset_type] = None

            # Read Fréchet Inception Distance (FID)
            try:
                with open(fid_txt_path, "r") as f:
                    lines = f.readlines()
                    fid_score = float(lines[0].strip().split()[-1])
                    scores["FID"][dataset_type] = fid_score
            except FileNotFoundError:
                print(f"Warning: FID score file {fid_txt_path} not found.")
                scores["FID"][dataset_type] = None

            # Read CLIP Score
            try:
                with open(clip_txt_path, "r") as f:
                    lines = f.readlines()
                    clip_score = float(lines[0].strip().split()[-1])
                    scores["CLIP"][dataset_type] = clip_score
            except FileNotFoundError:
                print(f"Warning: CLIP score file {clip_txt_path} not found.")
                scores["CLIP"][dataset_type] = None

        # Logging scores to WandB in a way that groups all datasets for each score type
        wandb.log({
            "IS": {"all": scores["IS"]["all"], "seen": scores["IS"]["seen"], "unseen": scores["IS"]["unseen"]},
            "FID": {"all": scores["FID"]["all"], "seen": scores["FID"]["seen"], "unseen": scores["FID"]["unseen"]},
            "CLIP": {"all": scores["CLIP"]["all"], "seen": scores["CLIP"]["seen"], "unseen": scores["CLIP"]["unseen"]}
        }, step=global_step)

        # Print scores for verification
        for score_type in scores:
            print(f"{score_type} Scores: {scores[score_type]}") 

        try:
            shutil.rmtree(args.save_dir)
            print(f"All folders in {args.save_dir} have been deleted.")
        except Exception as e:
            print(f"Error occurred while deleting folders in {args.save_dir}: {e}")
    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()