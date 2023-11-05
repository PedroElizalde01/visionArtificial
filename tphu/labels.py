# Define a mapping from string labels to integer labels
label_to_int_mapping = {
    'circle': 1,
    'square': 2,
    'triangle': 3,
}

# Define a mapping from integer labels to string labels
int_to_label_mapping = {
    1: 'circle',
    2: 'square',
    3: 'triangle',
}

def label_to_int(string_label):
    if string_label in label_to_int_mapping:
        return label_to_int_mapping[string_label]
    else:
        raise Exception('Unknown class_label')

def int_to_label(int_label):
    if int_label in int_to_label_mapping:
        return int_to_label_mapping[int_label]
    else:
        raise Exception('Unknown class_label')
