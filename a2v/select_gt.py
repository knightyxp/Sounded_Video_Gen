from glob import glob 
import os 


root = '/mnt/ssd4/yazhou'
ours_root = os.path.join(root, 'a2v-ours')
org_root = os.path.join(root, 'a2v-org')

# get all prompt and store them to a list 
all_files = sorted(glob(os.path.join(org_root, '*gif')))
prompts = []
for file in all_files:
    prompts.append(file.split('_')[1])

source_prompt = {}
source_root = '/home/yazhou/disk1/projects/edit/others/blip2/key_frames_VGGSound_3k'
txt_files = sorted(glob(os.path.join(source_root, '*txt')))

for t_file in txt_files:
    file_name = os.path.basename(t_file)
    with open(t_file, 'r') as f_open:
        line = f_open.readline()
        line = line.strip()
        print(line)
        source_prompt[line] = file_name

out_root = os.path.join(root, 'a2v-gt') 
os.makedirs(out_root, exist_ok=True)
source_videos_root = '/disk1/zeyue/project/data/vggsound/vggsound_3k_1122'
source_videos  = sorted(glob(os.path.join(source_videos_root, '*mp4'))) 
import shutil
for prompt in prompts:
    file_name = source_prompt[prompt]
    file_name = file_name.split('.')[0][:-2]
    mp4_name  = file_name + '.mp4'
    shutil.copy(os.path.join(source_videos_root, mp4_name), os.path.join(out_root, file_name+'_'+prompt+'.mp4'))
    print(os.path.join(out_root, file_name+'_'+prompt+'.mp4'))




