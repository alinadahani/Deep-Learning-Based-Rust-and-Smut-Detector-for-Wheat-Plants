from ultralytics import YOLO

# Load the models
model_1 = YOLO('D:/FYP Website - new version/wheaties/models/healthvsdiseased.pt')
model_2 = YOLO('D:/FYP Website - new version/wheaties/models/smutvsrust.pt')
model_3 = YOLO('D:/FYP Website - new version/wheaties/models/brownvsstem.pt')
model_4 = YOLO('D:/FYP Website - new version/wheaties/models/loose smut scoring.pt')
model_5 = YOLO('D:/FYP Website - new version/wheaties/models/brown rust scoring mode.pt')
model_6 = YOLO('D:/FYP Website - new version/wheaties/models/stemrust scoring.pt')

# Model selection dictionary
model_dict = {
    '1': model_1,
    '2': model_2,
    '3': model_3,
    '4': model_4,
    '5': model_5,
    '6': model_6
}
selected_option = input('select a model:')
selected_model = model_dict.get(selected_option)
if (selected_model.key()) > 4:
    label_counts = {label: 0 for label in selected_model.names.values()}
else:
    label_counts = {'low1':1 , 'med2': 2, 'high3': 3}
print(label_counts)
