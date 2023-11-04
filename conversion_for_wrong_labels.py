import os
import numpy as np

bad_labels = []
correct_labels = []

wrong_classes = '/home/jakub/sese_dataset/classes.txt'

with open(wrong_classes) as f:
    bad_labels = f.readlines()
    for i, label in enumerate(bad_labels):
        bad_labels[i] = label.rstrip()
print(bad_labels)

correct_classes = '/home/jakub/Downloads/predefined_classes.txt'

with open(correct_classes) as f:
    correct_labels = f.readlines()
    for i, label in enumerate(correct_labels):
        correct_labels[i] = label.rstrip()
print(correct_labels)

for file in os.listdir('/home/jakub/sese_dataset'):
    if file != 'classes.txt' and file != 'corrected':
        print(file)
        with open(os.path.join('/home/jakub/sese_dataset', file)) as f:
            lines = f.readlines()
            with open(os.path.join('/home/jakub/sese_dataset/corrected', file), 'w') as new_file:
                for line in lines:
                    file_class = bad_labels[int(line[0])]
                    try:
                        new_class = correct_labels.index(file_class)
                        new_file.write(f'{new_class}{line[1:]}')
                        print(file_class)
                        print(new_class)
                    except:
                        pass
