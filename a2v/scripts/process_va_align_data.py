import os, pipes

root= "/disk1/yingqing/disk2/code/AnimateDiff-Audio2Video/results/5-RealisticVision-abspath-2023-11-11T03-13-17-seed-1111-optimstep-1-with_text_loss"
ori_dir=os.path.join(root, "ori")
our_dir=os.path.join(root, "with_bind")

res_dir_ori="/disk1/yingqing/disk2/code/AnimateDiff-Audio2Video/results/5-RealisticVision-abspath-2023-11-11T03-13-17-seed-1111-optimstep-1-with_text_loss/sort-res/ori/"
os.makedirs(res_dir_ori, exist_ok=True)
res_dir_our="/disk1/yingqing/disk2/code/AnimateDiff-Audio2Video/results/5-RealisticVision-abspath-2023-11-11T03-13-17-seed-1111-optimstep-1-with_text_loss/sort-res/ours/"
os.makedirs(res_dir_our, exist_ok=True)

for fn in os.listdir(ori_dir):
    if "withaudio" in fn:
        fp = os.path.join(ori_dir, fn)
        os.system(f"cp " + pipes.quote(fp) + " " + res_dir_ori)
for fn in os.listdir(our_dir):
    if "withaudio" in fn:
        fp = os.path.join(our_dir, fn)
        os.system(f"cp " + pipes.quote(fp) + " " + res_dir_our)
