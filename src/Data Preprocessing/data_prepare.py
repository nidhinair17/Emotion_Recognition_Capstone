import os, shutil
import pandas as pd
data_folder = '../Data'
dataset_folder = "C:/Nini/Capstone/Dataset"
print(os.listdir(data_folder))

neutral_count = 0
calm_count = 0
happy_count = 0
sad_count = 0
angry_count = 0
fearful_count = 0
disgust_count = 0
surprised_count = 0
table = []
table.append(['path','source','gender','emotion','emotion_lb'])
for folder in os.listdir(data_folder):
    folder_path = f'{data_folder}/{folder}'
    for file in os.listdir(folder_path):
        file_split = file.split('-')
        print(file_split)
        if file_split[0] == '01':
            lst = []
            if file_split[2] == '01':
                os.makedirs(f'{dataset_folder}/Neutral',exist_ok=True)
                shutil.copyfile(f'{folder_path}/{file}',f'{dataset_folder}/Neutral/{file}')
                lst.append(f'{dataset_folder}/Neutral/{file}')
                lst.append('Ravdess')
                if int(file_split[6].split('.')[0]) % 2 == 0:
                    lst.append('female')
                else:
                    lst.append('male')
                lst.append(1)
                lst.append('neutral')
                table.append(lst)
                neutral_count += 1
            elif file_split[2] == '02':
                os.makedirs(f'{dataset_folder}/Calm',exist_ok=True)
                shutil.copyfile(f'{folder_path}/{file}',f'{dataset_folder}/Calm/{file}')
                lst.append(f'{dataset_folder}/Calm/{file}')
                lst.append('Ravdess')
                if int(file_split[6].split('.')[0]) % 2 == 0:
                    lst.append('female')
                else:
                    lst.append('male')
                lst.append(2)
                lst.append('calm')
                table.append(lst)
                calm_count += 1
            elif file_split[2] == '03':
                os.makedirs(f'{dataset_folder}/Happy',exist_ok=True)
                shutil.copyfile(f'{folder_path}/{file}',f'{dataset_folder}/Happy/{file}')
                lst.append(f'{dataset_folder}/Happy/{file}')
                lst.append('Ravdess')
                if int(file_split[6].split('.')[0]) % 2 == 0:
                    lst.append('female')
                else:
                    lst.append('male')
                lst.append(3)
                lst.append('happy')
                table.append(lst)
                happy_count += 1
            elif file_split[2] == '04':
                os.makedirs(f'{dataset_folder}/Sad',exist_ok=True)
                shutil.copyfile(f'{folder_path}/{file}',f'{dataset_folder}/Sad/{file}')
                lst.append(f'{dataset_folder}/Sad/{file}')
                lst.append('Ravdess')
                if int(file_split[6].split('.')[0]) % 2 == 0:
                    lst.append('female')
                else:
                    lst.append('male')
                lst.append(4)
                lst.append('sad')
                table.append(lst)
                sad_count += 1
            elif file_split[2] == '05':
                os.makedirs(f'{dataset_folder}/Angry',exist_ok=True)
                shutil.copyfile(f'{folder_path}/{file}',f'{dataset_folder}/Angry/{file}')
                lst.append(f'{dataset_folder}/Angry/{file}')
                lst.append('Ravdess')
                if int(file_split[6].split('.')[0]) % 2 == 0:
                    lst.append('female')
                else:
                    lst.append('male')
                lst.append(5)
                lst.append('angry')
                table.append(lst)
                angry_count += 1
            elif file_split[2] == '06':
                os.makedirs(f'{dataset_folder}/Fearful',exist_ok=True)
                shutil.copyfile(f'{folder_path}/{file}',f'{dataset_folder}/Fearful/{file}')
                lst.append(f'{dataset_folder}/Fearful/{file}')
                lst.append('Ravdess')
                if int(file_split[6].split('.')[0]) % 2 == 0:
                    lst.append('female')
                else:
                    lst.append('male')
                lst.append(6)
                lst.append('fearful')
                table.append(lst)
                fearful_count += 1
            elif file_split[2] == '07':
                os.makedirs(f'{dataset_folder}/Disgust',exist_ok=True)
                shutil.copyfile(f'{folder_path}/{file}',f'{dataset_folder}/Disgust/{file}')
                lst.append(f'{dataset_folder}/Disgust/{file}')
                lst.append('Ravdess')
                if int(file_split[6].split('.')[0]) % 2 == 0:
                    lst.append('female')
                else:
                    lst.append('male')
                lst.append(7)
                lst.append('disgust')
                table.append(lst)
                disgust_count += 1
            elif file_split[2] == '08':
                os.makedirs(f'{dataset_folder}/Surprised',exist_ok=True)
                shutil.copyfile(f'{folder_path}/{file}',f'{dataset_folder}/Surprised/{file}')
                lst.append(f'{dataset_folder}/Surprised/{file}')
                lst.append('Ravdess')
                if int(file_split[6].split('.')[0]) % 2 == 0:
                    lst.append('female')
                else:
                    lst.append('male')
                lst.append(8)
                lst.append('surprised')
                table.append(lst)
                surprised_count += 1
df = pd.DataFrame(table[1:], columns=table[0])
output_file_path = f'{dataset_folder}/Ravdess.csv'
df.to_csv(output_file_path,index=False)
print(f'Neutral: {neutral_count}')
print(f'Calm: {calm_count}')
print(f'Happy: {happy_count}')
print(f'sad: {sad_count}')
print(f'Angry: {angry_count}')
print(f'Fearful: {fearful_count}')
print(f'Disgust: {disgust_count}')
print(f'Surprised: {surprised_count}')

print(table)