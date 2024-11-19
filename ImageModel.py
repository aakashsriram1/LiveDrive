import os
import pandas as pd
import time

# Specify the path to the directory containing the screenshots
directory_path = "/Users/ryanbacich/Desktop/Cropped Image Data"

play_data = pd.read_csv("Final_Combined_Play_Data_Live Drive.csv")
count = play_data['offenseFormation'].value_counts()
print(count)

 
#time.sleep(1000)
#### GAME 6 DATA IS UNCOMPLETE**********
#### GAME 4 DATA IS UNCOMPLETE**********

# List all files in the directory
file_names = os.listdir(directory_path)

# Initialize lists to hold gameId and playId
game_ids = []
play_ids = []

# Loop through the filenames and parse them
for file_name in file_names:
    if file_name.endswith('.PNG'):  # Check if the file is a PNG image
        base_name = os.path.splitext(file_name)[0]  # Remove the file extension
        parts = base_name.split('-')  # Split the filename by '-'
        if len(parts) == 2:  # Ensure the filename is in the expected format
            game_ids.append(parts[0])
            play_ids.append(parts[1])

# Create a DataFrame from the gameId and playId
df = pd.DataFrame({
    'gameId': game_ids,
    'playId': play_ids
})

# Save the DataFrame to a CSV file
csv_file_path = 'ImageOffensiveFormations12.csv'  # Specify your output CSV file name
df.to_csv(csv_file_path, index=False)

print(f"CSV file has been created and saved to {csv_file_path}")

#time.sleep(100000) 

 #FINAL TRAIN FILE FOR PLAYS ACTIONS
# Load the original dataset and the predictions
pass_plays_data = pd.read_csv('Final_2021_Pass_Play_Data.csv')
run_pass_plays_data = pd.read_csv("Final_2022_Run_Pass_Play_Data.csv")
combined_play_data = pd.concat([pass_plays_data, run_pass_plays_data], ignore_index=True)
combined_play_data = combined_play_data.drop_duplicates(subset=['gameId', 'playId'], keep='first')
print(len(combined_play_data), "len play")
combined_play_data = combined_play_data.dropna(subset=['offenseFormation'])
combined_play_data = combined_play_data
combined_play_data.to_csv("Final_Combined_Play_Data_Live Drive.csv", index = False)
import pandas as pd

# Load the CSV file into a DataFrame
image_formations_df = pd.read_csv('ImageOffensiveFormations12.csv')

# Assume combined_play_data is a DataFrame that's already in your environment
# with the necessary columns, including 'OffenseFormation'.
# If not, you'd need to load it from a CSV or another source like this:
# combined_play_data = pd.read_csv('combined_play_data.csv')

# Merge to add the 'OffenseFormation' column to the image formations DataFrame
merged_df = image_formations_df.merge(combined_play_data[['gameId', 'playId', 'offenseFormation']],
                                      on=['gameId', 'playId'],
                                      how='left')

# Save the updated DataFrame back to a CSV file
merged_df.to_csv('ImageOffensiveFormations_with_Formation12.csv', index=False)

print("The 'OffenseFormation' column has been added and saved to 'ImageOffensiveFormations_with_Formation11.csv'")

#time.sleep(100000)

#### IMAGE OFFENSIVE FORMATION MODEL.


import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, precision_score

# Load and map labels to images
csv_file = "ImageOffensiveFormations_with_Formation12.csv"
labels = pd.read_csv(csv_file)

# Adjust classes
labels = labels[labels['offenseFormation'] != 'WILDCAT']
labels = labels[labels['offenseFormation'] != 'JUMBO']  # Remove problematic classes
print("Summary of the 'offenseFormation' column:")
print(labels['offenseFormation'].value_counts())

# Generate 'image_filename' and 'image_path'
labels['image_filename'] = labels.apply(lambda r: f"{r['gameId']}-{r['playId']}.PNG", axis=1)
image_dir = "/Users/ryanbacich/Desktop/Cropped Image Data"
labels['image_path'] = labels['image_filename'].apply(lambda x: os.path.join(image_dir, x))

# Ensure there are no NaN values in the 'offenseFormation' column
labels = labels.dropna(subset=['offenseFormation'])

# Split the data
train_data, test_data = train_test_split(labels, test_size=0.2, random_state=42, stratify=labels['offenseFormation'])
train_data, validate_data = train_test_split(train_data, test_size=0.25, random_state=42, stratify=train_data['offenseFormation'])

print(train_data['offenseFormation'].value_counts())
print(validate_data['offenseFormation'].value_counts())

print(test_data['offenseFormation'].value_counts())



class_weight = {'EMPTY': 3.410989010989011,
 'I_FORM': 2.544262295081967,
 'PISTOL': 4.252054794520548,
 'SHOTGUN': 0.366903073286052,
 'SINGLEBACK': 0.7390476190476191}




new_width = 546  # Example reduced width
new_height = 310

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


# Prepare data generators
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_data,
    directory=image_dir,
    x_col='image_path',
    y_col='offenseFormation',
    target_size=(new_height, new_width),
    class_mode='categorical',
    batch_size=32
)
validation_generator = test_datagen.flow_from_dataframe(
    dataframe=validate_data,
    directory=image_dir,
    x_col='image_path',
    y_col='offenseFormation',
    target_size=(new_height, new_width),
    class_mode='categorical',
    batch_size=32
)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_data,
    directory=image_dir,
    x_col='image_path',
    y_col='offenseFormation',
    target_size=(new_height, new_width),
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)
# Model building



base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(new_height, new_width, 3))

# Make adjustments to the model
for layer in base_model.layers[:-15]:
    layer.trainable = False

# Continue building the model as before
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(len(np.unique(train_data['offenseFormation'])), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20,
    class_weight=class_weight,  # add the class_weight here
    callbacks=[early_stopping]
)


# Evaluate and report performance
results = model.evaluate(test_generator)
print("Test loss, Test accuracy:", results)

test_generator.reset()
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_generator.classes
predictions_df = pd.DataFrame({
    'gameId': test_data['gameId'].reset_index(drop=True),
    'playId': test_data['playId'].reset_index(drop=True),
    'PredictedFormation': [list(train_generator.class_indices)[p] for p in y_pred]
})
predictions_df.to_csv('Predictions.csv', index=False)

# Confusion matrix and class-specific accuracy
cm = confusion_matrix(y_true, y_pred)
print('Confusion Matrix')
print(cm)
class_accuracy = 100 * cm.diagonal() / cm.sum(axis=1)
print('Class Accuracy :')
for i, accuracy in enumerate(class_accuracy):
    print(f"{list(train_generator.class_indices)[i]}: {accuracy:.2f}%")

# Precision
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
print(f"Model Precision: {precision}")

# Save the model in the new format
model.save('nfl_offensive_formation_classifier.keras') 



import tensorflow as tf
import watchdog
from watchdog.events import FileSystemEventHandler
import os
import time
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
from watchdog.observers import Observer


class_labels_dict = {0: 'EMPTY', 1: 'I-FORM', 2: 'PISTOL', 3: 'SHOTGUN', 4: 'SINGLEBACK'}  
model = tf.keras.models.load_model('nfl_offensive_formation_classifier.keras')

# Set the directory to watch for new screenshots
watch_directory = "/Users/ryanbacich/Desktop/LiveDrive/LiveDriveData/LiveDriveScreenshotTest"

Final_Play_Data = pd.read_csv("Final_Combined_Play_Data_Live Drive.csv")

class ScreenshotHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            # Extract the filename from the full path
            filename = os.path.basename(event.src_path)
            # Check if the file name starts with 'Screenshot' (case-insensitive)
            if filename.lower().startswith("screenshot"):
                # Give some time for the file to finish writing
                time.sleep(0.3)
                self.process_screenshot(event.src_path)

    def process_screenshot(self, filepath):
        try:
            print(f"Processing new screenshot: {filepath}")
            img = keras_image.load_img(filepath, target_size=(310, 546))
            img_array = keras_image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions, axis=1)
            predicted_class = class_labels_dict[int(predicted_class_index[0])]
            print(f"Predicted offensive formation: {predicted_class}")
            
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")

def start_monitoring(directory):
    event_handler = ScreenshotHandler()
    observer = Observer()
    observer.schedule(event_handler, directory, recursive=True) # Set recursive=True to monitor subdirectories
    observer.start()
    print(f"Watching for new screenshots in {directory}...")

    try:
        while True:
            time.sleep(0.3)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    start_monitoring(watch_directory)
 
#time.sleep(10000)


#### IMAGE CLASSIFICATION MODEL














#TRACKING TO OFFENSIVE FORMATION MODEL

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
import json

import pandas as pd
# Load the new data
file_path = 'offense_only_2021_tracking_passing.csv'

new_data = pd.read_csv(file_path)
class MultiLabelBinarizerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = MultiLabelBinarizer()
    
    def fit(self, X, y=None):
        # Assume that X is a DataFrame that contains one column of lists
        self.mlb.fit(X.iloc[:, 0])
        return self
    
    def transform(self, X):
        result = self.mlb.transform(X.iloc[:, 0])
        df_result = pd.DataFrame(result, columns=self.mlb.classes_)
        #Index aligns with the input DF
        df_result.index = X.index
        return df_result


#(70% training, 30% testing)
new_train, new_test = train_test_split(new_data, test_size=0.3, random_state=42)


def load_and_preprocess_data(week_numbers):
    dfs = []
    for i in week_numbers:
        file_path = f"offense_only_final_data_week{i}_2024.csv"
        df_week = pd.read_csv(file_path)
        df_week['mean_distance'] = df_week.groupby(['gameId', 'playId'])['dis'].transform('mean')
        #df_week['spread_distance'] = df_week.groupby(['gameId', 'playId'])['dis'].transform('std')
        df_week['max_distance'] = df_week.groupby(['gameId', 'playId'])['dis'].transform('max')
        df_week['centroid_x'] = df_week.groupby(['gameId', 'playId'])['x'].transform('mean')
        df_week['centroid_y'] = df_week.groupby(['gameId', 'playId'])['y'].transform('mean')
        df_week['spread_x'] = df_week.groupby(['gameId', 'playId'])['x'].transform('std')
        df_week['spread_y'] = df_week.groupby(['gameId', 'playId'])['y'].transform('std')
        df_week['max_speed'] = df_week.groupby(['gameId', 'playId'])['s'].transform('max')
        df_week['max_acceleration'] = df_week.groupby(['gameId', 'playId'])['a'].transform('max')
        df_week['average_orientation'] = df_week.groupby(['gameId', 'playId'])['o'].transform('mean')
        df_week['average_direction'] = df_week.groupby(['gameId', 'playId'])['dir'].transform('mean')


        qb_distances = df_week[df_week['player_position'] == 'QB'][['gameId', 'playId', 'dis']]
        qb_distances.rename(columns={'dis': 'qb_dis'}, inplace=True)
        df_week = df_week.merge(qb_distances, on=['gameId', 'playId'], how='left')
        df_week['Distance-To-QB'] = (df_week['qb_dis'])
        df_week = df_week.drop(['qb_dis'], axis=1)

        rb_df = df_week[df_week['player_position'] == 'RB']
        qb_df = df_week[df_week['player_position'] == 'QB']

        # Merge the RB and QB dataframes on gameId and playId to calc distances
        rb_qb_merged = pd.merge(rb_df, qb_df, on=['gameId', 'playId'], suffixes=('_rb', '_qb'))
        
        # Calculate distance between RB and QB for each play
        rb_qb_merged['rb_qb_distance'] = np.sqrt(
            (rb_qb_merged['x_rb'] - rb_qb_merged['x_qb'])**2 + (rb_qb_merged['y_rb'] - rb_qb_merged['y_qb'])**2
        )
        # Merge the distance calculation back to the original dataframe
        df_week = pd.merge(df_week, rb_qb_merged[['gameId', 'playId', 'rb_qb_distance']], on=['gameId', 'playId'], how='left')

        positions_list = df_week.groupby(['gameId', 'playId'])['player_position'].agg(lambda x: sorted(list(x))).reset_index()
        positions_list.rename(columns={'player_position': 'positions_list'}, inplace=True)
        df_week = df_week.drop("player_position", axis=1)
        df_week = df_week.drop("o", axis=1)
        df_week = df_week.drop("dir", axis=1)
        #print(positions_list, "POS LIST")
        # Merge positions list with the main DataFrame
        df_week = df_week.drop_duplicates(subset=['gameId', 'playId'])
        df_week = pd.merge(df_week, positions_list, on=['gameId', 'playId'], how='left')

        dfs.append(df_week)
        
    df_combined = pd.concat(dfs, ignore_index=True)
    return df_combined


train_weeks = range(1, 5)
test_weeks = range(5, 8)

new_train['mean_distance'] = new_train.groupby(['gameId', 'playId'])['dis'].transform('mean')
#new_train['spread_distance'] = new_train.groupby(['gameId', 'playId'])['dis'].transform('std')
new_train['max_distance'] = new_train.groupby(['gameId', 'playId'])['dis'].transform('max')
new_train['centroid_x'] = new_train.groupby(['gameId', 'playId'])['x'].transform('mean')
new_train['centroid_y'] = new_train.groupby(['gameId', 'playId'])['y'].transform('mean')
new_train['spread_x'] = new_train.groupby(['gameId', 'playId'])['x'].transform('std')
new_train['spread_y'] = new_train.groupby(['gameId', 'playId'])['y'].transform('std')
new_train['max_speed'] = new_train.groupby(['gameId', 'playId'])['s'].transform('max')
new_train['max_acceleration'] = new_train.groupby(['gameId', 'playId'])['a'].transform('max')
new_train['average_orientation'] = new_train.groupby(['gameId', 'playId'])['o'].transform('mean')
new_train['average_direction'] = new_train.groupby(['gameId', 'playId'])['dir'].transform('mean')


rb_df = new_train[new_train['player_position'] == 'RB']
qb_df = new_train[new_train['player_position'] == 'QB']

# Merge RB and QB dataframes on gameId and playId to calc distances
rb_qb_merged = pd.merge(rb_df, qb_df, on=['gameId', 'playId'], suffixes=('_rb', '_qb'))

# Calc distance between RB and QB for each play
rb_qb_merged['rb_qb_distance'] = np.sqrt(
    (rb_qb_merged['x_rb'] - rb_qb_merged['x_qb'])**2 + (rb_qb_merged['y_rb'] - rb_qb_merged['y_qb'])**2
)

# Merge the distance calc back to the original dataframe
new_train = pd.merge(new_train, rb_qb_merged[['gameId', 'playId', 'rb_qb_distance']], on=['gameId', 'playId'], how='left')


qb_distances = new_train[new_train['player_position'] == 'QB'][['gameId', 'playId', 'dis']]
qb_distances.rename(columns={'dis': 'qb_dis'}, inplace=True)
new_train = new_train.merge(qb_distances, on=['gameId', 'playId'], how='left')
new_train['Distance-To-QB'] = (new_train['qb_dis'])
new_train = new_train.drop(['qb_dis'], axis=1)

new_train = new_train.drop_duplicates(subset=['gameId', 'playId'])

positions_list = new_train.groupby(['gameId', 'playId'])['player_position'].agg(lambda x: sorted(list(x))).reset_index()
positions_list.rename(columns={'player_position': 'positions_list'}, inplace=True)
new_train = new_train.drop("player_position", axis=1)
new_train = new_train.drop("o", axis=1)
new_train = new_train.drop("dir", axis=1)
print(positions_list, "POS LIST")
# Merge the positions list with the main DF
new_train = new_train.drop_duplicates(subset=['gameId', 'playId'])
new_train = pd.merge(new_train, positions_list, on=['gameId', 'playId'], how='left')

new_test['mean_distance'] = new_test.groupby(['gameId', 'playId'])['dis'].transform('mean')
#new_test['spread_distance'] = new_test.groupby(['gameId', 'playId'])['dis'].transform('std')
new_test['max_distance'] = new_test.groupby(['gameId', 'playId'])['dis'].transform('max')
new_test['centroid_x'] = new_test.groupby(['gameId', 'playId'])['x'].transform('mean')
new_test['centroid_y'] = new_test.groupby(['gameId', 'playId'])['y'].transform('mean')
new_test['spread_x'] = new_test.groupby(['gameId', 'playId'])['x'].transform('std')
new_test['spread_y'] = new_test.groupby(['gameId', 'playId'])['y'].transform('std')
new_test['max_speed'] = new_test.groupby(['gameId', 'playId'])['s'].transform('max')
new_test['max_acceleration'] = new_test.groupby(['gameId', 'playId'])['a'].transform('max')
new_test['average_orientation'] = new_test.groupby(['gameId', 'playId'])['o'].transform('mean')
new_test['average_direction'] = new_test.groupby(['gameId', 'playId'])['dir'].transform('mean')


rb_df = new_test[new_test['player_position'] == 'RB']
qb_df = new_test[new_test['player_position'] == 'QB']

# Merge RB and QB dataframes on gameId and playId to calc distances
rb_qb_merged = pd.merge(rb_df, qb_df, on=['gameId', 'playId'], suffixes=('_rb', '_qb'))

# Calculate the distance between RB and QB for each play
rb_qb_merged['rb_qb_distance'] = np.sqrt(
    (rb_qb_merged['x_rb'] - rb_qb_merged['x_qb'])**2 + (rb_qb_merged['y_rb'] - rb_qb_merged['y_qb'])**2
)
# Merge distance calc back to the original dataframe
new_test = pd.merge(new_test, rb_qb_merged[['gameId', 'playId', 'rb_qb_distance']], on=['gameId', 'playId'], how='left')

qb_distances = new_test[new_test['player_position'] == 'QB'][['gameId', 'playId', 'dis']]
qb_distances.rename(columns={'dis': 'qb_dis'}, inplace=True)
new_test = new_test.merge(qb_distances, on=['gameId', 'playId'], how='left')
new_test['Distance-To-QB'] = (new_test['qb_dis'])
new_test = new_test.drop(['qb_dis'], axis=1)

new_test = new_test.drop_duplicates(subset=['gameId', 'playId'])

positions_list = new_test.groupby(['gameId', 'playId'])['player_position'].agg(lambda x: sorted(list(x))).reset_index()

positions_list.rename(columns={'player_position': 'positions_list'}, inplace=True)
new_test = new_test.drop("player_position", axis=1)
new_test = new_test.drop("o", axis=1)
new_test = new_test.drop("dir", axis=1)
print(positions_list, "POS LIST")
# Merge the positions list with the main DataFrame
new_test = new_test.drop_duplicates(subset=['gameId', 'playId'])
new_test = pd.merge(new_test, positions_list, on=['gameId', 'playId'], how='left')

new_train = new_train.dropna(subset=['offensive_formation'])
new_test = new_test.dropna(subset=['offensive_formation'])
new_train = new_train[new_train['offensive_formation'] != 'WILDCAT']
new_test = new_test[new_test['offensive_formation'] != 'WILDCAT']

df_train = load_and_preprocess_data(train_weeks)
df_test = load_and_preprocess_data(test_weeks)

df_train = df_train.dropna(subset=['offensive_formation'])
df_test = df_test.dropna(subset=['offensive_formation'])
df_test = df_test[df_test['offensive_formation'] != 'WILDCAT']
df_train = df_train[df_train['offensive_formation'] != 'WILDCAT']

combined_train = pd.concat([df_train, new_train], ignore_index=True)
combined_test = pd.concat([df_test, new_test], ignore_index=True)
combined_train.drop(['displayName', 'jerseyNumber'], axis=1)
combined_test.drop(['displayName', 'jerseyNumber'], axis=1)
# Prepare features and target variables
y_train = combined_train['offensive_formation']
y_test = combined_test['offensive_formation']
X_train = combined_train.drop(['offensive_formation', "dis", 'x', 'y', 's', 'a', 'gameId', 'playId', 'nflId', 'jerseyNumber', "displayName", "teamType"], axis=1)
X_test = combined_test.drop(['offensive_formation', "dis", 'x', 'y', 's', 'a', 'gameId', 'playId', 'nflId', 'jerseyNumber', "displayName", "teamType"], axis=1)
print("Length of X_train:", len(X_train))
print("Length of y_train:", len(y_train))
print("Length of X_test:", len(X_test))
print("Length of y_test:", len(y_test))

# Encode categorical variables and scale numerical features
categorical_features = [col for col in X_train.select_dtypes(include=['object']).columns if col != 'positions_list' and col != 'player_position']
print("categorical_features", categorical_features)
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
print("numerical_features", numerical_features)

multi_label_feature = ['positions_list']

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Impute NaNs
    ('scaler', StandardScaler())])

multi_label_transformer = Pipeline(steps=[
    ('mlb', MultiLabelBinarizerWrapper())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        ('multi_label', multi_label_transformer, multi_label_feature)
    ])

# Define the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=1000, random_state=42))
])

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Combine gameId, playId with predictions into a DataFrame
predictions_df = pd.DataFrame({
    'gameId': combined_test['gameId'],
    'playId': combined_test['playId'],
    'predictedOffensiveFormation': y_pred
})

# Save the Offensive Formation predictions to a CSV file
predictions_df.to_csv('predicted_offensive_formations.csv', index=False)

accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy on Test Data Tracking_Offense_Formation: {accuracy}')
report = classification_report(y_test, y_pred, target_names=np.unique(y_test), output_dict=True)
print("Detailed Classification Report Tracking_Offense_Formation:")
print(classification_report(y_test, y_pred, target_names=np.unique(y_test)))


print("Confusion Matrix Tracking_Offense_Formation:")
print(confusion_matrix(y_test, y_pred))


with open('classification_report_Tracking_Offense_Formation.json', 'w') as f:
    json.dump(report, f)


cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=np.unique(y_test), columns=np.unique(y_test))
cm_df.to_csv('confusion_matrix_Tracking_Offense_Formation.csv')

#time.sleep(100000)




#END OF TRACKING TO OFFENSR FORMATION MODERL
""" 
file_path = 'offense_only_2021_tracking_passing.csv'
new_data = pd.read_csv(file_path)
new_data.rename(columns={'team': 'club'}, inplace=True)
new_data.rename(columns={'offenseFormation': 'offensive_formation'}, inplace=True)
new_data.to_csv("offense_only_2021_tracking_passing.csv", index = False)
print('done') 
time.sleep(1000)
import pandas as pd 
"""

# Load the datasets
""" plays_data = pd.read_csv('2021-plays.csv')
pass_data = pd.read_csv('OFFENSE_2021_combined_ball_snap.csv')

# Check and convert data types if necessary
plays_data['gameId'] = plays_data['gameId'].astype(int)
plays_data['playId'] = plays_data['playId'].astype(int)
pass_data['gameId'] = pass_data['gameId'].astype(int)
pass_data['playId'] = pass_data['playId'].astype(int)

# Merge the datasets
# Ensure that only 'playDirection' is added from the pass_data
merged_data = pd.merge(plays_data, pass_data[['gameId', 'playId', 'playDirection']], on=['gameId', 'playId'], how='left')

df = pd.read_csv('2021-plays.csv')
merged_data.rename(columns={'defendersInBox': 'defendersInTheBox'}, inplace=True)
merged_data['PlayType'] = "Pass"



columns_to_keep = [
    "gameId", "playId", 'quarter', 'down', 'yardsToGo', 'possessionTeam', 'defensiveTeam', 
    'yardlineSide', 'yardlineNumber', 'gameClock', 'preSnapHomeScore', 
    'preSnapVisitorScore', 'defendersInTheBox', 'offenseFormation', 
    'PlayType', "playDescription", "playDirection"
]
merged_data = merged_data[columns_to_keep]

# Save the updated DataFrame
merged_data.to_csv('aoffense_only_2021_tracking_passing.csv', index=False) """

######MERGE THE CORROPSONDING PLAYS
import pandas as pd

# Load the original dataset and the predictions
pass_plays_data = pd.read_csv('Final_2021_Pass_Play_Data.csv')
run_pass_plays_data = pd.read_csv("Final_2022_Run_Pass_Play_Data.csv")
original_data = pd.concat([pass_plays_data, run_pass_plays_data], ignore_index=True)
print(len(original_data), "rows in original data")
original_data = original_data.drop_duplicates(subset=['gameId', 'playId'], keep='first')
print(len(original_data), "rows in original data next")
predicted_tracking_formations = pd.read_csv('predicted_offensive_formations.csv')
predicted_image_formations = pd.read_csv("Predictions.csv")
predicted_formations = pd.concat([predicted_tracking_formations, predicted_image_formations], ignore_index=True)
predicted_formations = predicted_formations.drop_duplicates(subset=['gameId', 'playId'], keep='first')
print(len(predicted_formations), "LEN PREDICTED")

# Ensure the key columns are of the same type, typically int
original_data['gameId'] = original_data['gameId'].astype(int)
original_data['playId'] = original_data['playId'].astype(int)
predicted_formations['gameId'] = predicted_formations['gameId'].astype(int)
predicted_formations['playId'] = predicted_formations['playId'].astype(int)

# Merge the datasets using an inner join to retain only matched rows
matched_data = original_data.merge(predicted_formations[['gameId', 'playId', 'predictedOffensiveFormation']],
                                   on=['gameId', 'playId'], 
                                   how='inner')

# Replace 'offenseFormation' with 'predictedOffensiveFormation'
matched_data['offenseFormation'] = matched_data['predictedOffensiveFormation']

# Drop the temporary 'predictedOffensiveFormation' column now that its data has been used
matched_data.drop(columns=['predictedOffensiveFormation'], inplace=True)

# Remove duplicates while keeping the first occurrence
matched_data = matched_data.drop_duplicates(subset=['gameId', 'playId'], keep='first')

# Save the matched dataset to a new CSV file
matched_data.to_csv('FINAL_PREDICTIONS_TEST_FILE.csv', index=False)
print("Updated and matched data saved to 'FINAL_PREDICTIONS_TEST_FILE.csv'.")
print(len(matched_data), "rows in FINAL_PREDICTIONS_TEST_FILE data")


# Find unmatched rows using a full outer join and an indicator
full_merged = original_data.merge(predicted_formations[['gameId', 'playId', 'predictedOffensiveFormation']],
                                  on=['gameId', 'playId'], 
                                  how='outer', 
                                  indicator=True)

# Select only unmatched rows and ensure it is a copy to avoid SettingWithCopyWarning
unmatched_data = full_merged[full_merged['_merge'] != 'both'].copy()

# Drop the merge indicator column safely
unmatched_data.drop(columns=['_merge'], inplace=True)

# Optionally, remove duplicates again if needed (this depends on your specific case)
unmatched_data.drop_duplicates(subset=['gameId', 'playId'], inplace=True)

# Save the unmatched dataset to a new CSV file
unmatched_data.to_csv('FINAL_PLAY_TRAIN_DATA.csv', index=False)
print(len(unmatched_data), "rows in FINAL_PLAY_TRAIN_DATA")
print("Unmatched data saved to 'FINAL_PLAY_TRAIN_DATA.csv'.")



time.sleep(5)
######MERGE THE CORROPSONDING PLAYS
# Define the columns to keep



import pandas as pd



#############OFFENSE FORMATION MODEL







""" import pandas as pd

# Load both datasets
original_data = pd.read_csv('NEW_FINAL_SORTED_PLAYS_NEW.csv')
filtered_data = pd.read_csv('cleaned_updated_final_sorted_plays.csv')

# Ensure the key columns are the same type, usually integer
original_data['gameId'] = original_data['gameId'].astype(int)
original_data['playId'] = original_data['playId'].astype(int)
filtered_data['gameId'] = filtered_data['gameId'].astype(int)
filtered_data['playId'] = filtered_data['playId'].astype(int)

# Perform an anti-join to find rows in original_data not present in filtered_data
# This is done by merging with an indicator and then filtering
merged_data = pd.merge(original_data, filtered_data[['gameId', 'playId']],
                       on=['gameId', 'playId'], 
                       how='left', 
                       indicator=True)

# Filter out the rows that are found in both data sets
result_data = merged_data[merged_data['_merge'] == 'left_only'].drop('_merge', axis=1)

# Save the filtered data to a new CSV file
result_data.to_csv('FINAL_TRAINER.csv', index=False)
print("Filtered data saved to 'FINAL_TRAINER.csv'.") """



pass_plays_data = pd.read_csv('Final_2021_Pass_Play_Data.csv')
run_pass_plays_data = pd.read_csv("Final_2022_Run_Pass_Play_Data.csv")
original_data = pd.concat([pass_plays_data, run_pass_plays_data], ignore_index=True)


predicted_formations = pd.read_csv("Image_Model_OffenseFormation_Predictions.csv")

predicted_formations.rename(columns={'PredictedFormation': 'predictedOffensiveFormation'}, inplace=True)
# Ensure the key columns are of the same type, typically int
original_data['gameId'] = original_data['gameId'].astype(int)
original_data['playId'] = original_data['playId'].astype(int)
predicted_formations['gameId'] = predicted_formations['gameId'].astype(int)
predicted_formations['playId'] = predicted_formations['playId'].astype(int)





# Merge the datasets using an inner join to ensure only matched rows are retained
updated_data = original_data.merge(predicted_formations[['gameId', 'playId', 'predictedOffensiveFormation']],
                                   on=['gameId', 'playId'], 
                                   how='inner')

# Replace 'offenseFormation' with 'predictedOffensiveFormation'
updated_data['offenseFormation'] = updated_data['predictedOffensiveFormation']

# Drop the temporary 'predictedOffensiveFormation' column now that its data has been used
updated_data.drop(columns=['predictedOffensiveFormation'], inplace=True)

# Remove duplicates while keeping the first occurrence
updated_data = updated_data.drop_duplicates(subset=['gameId', 'playId'], keep='first')

# Save the updated dataset to a new CSV file
updated_data.to_csv('FINAL_IMAGE_PREDICTIONS_TRAIN_FILE.csv', index=False)
print("Updated and cleaned data saved to 'FINAL_PREDICTIONS_TRAIN_FILE.csv'.")

time.sleep(5)

test_file_path = "FINAL_IMAGE_PREDICTIONS_TRAIN_FILE.csv"




data = pd.read_csv('FINAL_PREDICTIONS_TEST_FILE.csv')

# Separate the data by class
pass_data = data[data['PlayType'] == 'Pass']
run_data = data[data['PlayType'] == 'Run']

# Sample 2401 instances from the "Pass" data
pass_data_sampled = pass_data.sample(n=2401, random_state=42)

# Concatenate the "Run" data with the sampled "Pass" data
balanced_data = pd.concat([pass_data_sampled, run_data])

# Optionally shuffle the dataset to mix pass and run plays
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the balanced dataset to a new CSV file
balanced_data.to_csv('ADJUSTED_FINAL_PREDICTIONS_TEST_FILE.csv', index=False)

print("Adjusted dataset saved with 2401 Pass plays and 1964 Run plays.")





##########
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib  # For saving the model and label encoders
import time
import numpy as np

def convert_time_to_seconds(time_str):
    minutes, seconds = map(int, time_str.split(':'))
    return minutes * 60 + seconds

def preprocess_data(file_path, label_encoders=None, fit_encoders=False):
    data = pd.read_csv(file_path)
    
    # Use ffill() to fill missing values forward
    data.ffill(inplace=True)
    
    data['gameClock'] = data['gameClock'].apply(convert_time_to_seconds)

    # Define categorical columns
    categorical_columns = ['possessionTeam', 'defensiveTeam', 'yardlineSide', 'offenseFormation', 'playDirection']
    
    if fit_encoders:
        label_encoders = {}
        for column in categorical_columns:
            le = LabelEncoder()
            data[column] = data[column].fillna('Unknown')  # Replace NaNs with 'Unknown' if any
            data[column] = le.fit_transform(data[column])
            label_encoders[column] = le
    else:
        for column in categorical_columns:
            data[column] = data[column].fillna('Unknown')
            data[column] = label_encoders[column].transform(data[column])

    return data, label_encoders

# Load and preprocess training data
train_data, label_encoders = preprocess_data('FINAL_PLAY_TRAIN_DATA.csv', fit_encoders=True)

# Manually specify class weights
class_weights_dict = {'Pass': 0.55, 'Run': 0.45}

# Features and target specification
features = ['quarter', 'down', 'yardsToGo', 'possessionTeam', 'defensiveTeam', 'yardlineSide',
            'yardlineNumber', 'gameClock', 'preSnapHomeScore', 'preSnapVisitorScore', 'offenseFormation', 'playDirection']
target = 'PlayType'

# Separate features and target for training
X_train = train_data[features]
y_train = train_data[target]


class_counts = y_train.value_counts()
class_percentages = y_train.value_counts(normalize=True) * 100  # normalize=True gives the relative frequencies

# Print the results
print("Counts of each class:")
print(class_counts)
print("\nPercentage of each class:")
print(class_percentages)

# Train the RandomForestClassifier with specified class weights
rf_classifier = RandomForestClassifier(n_estimators=500, max_depth=20, min_samples_split=10,
                                       class_weight=class_weights_dict, random_state=42)
rf_classifier.fit(X_train, y_train)

# Save the trained model and label encoders
joblib.dump(rf_classifier, 'rf_classifier.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')




# Load and preprocess test data
test_data, _ = preprocess_data('ADJUSTED_FINAL_PREDICTIONS_TEST_FILE.csv', label_encoders=label_encoders, fit_encoders=False)

# Separate features and target for testing
X_test = test_data[features]
y_test = test_data[target]

# Predict using the trained model
y_pred = rf_classifier.predict(X_test)

# Evaluate and print model performance on test data
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the model on new test data:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

time.sleep(5)

time.sleep(5)


# Load and preprocess test data
test_data, _ = preprocess_data('FINAL_IMAGE_PREDICTIONS_TRAIN_FILE.csv', label_encoders=label_encoders, fit_encoders=False)

# Separate features and target for testing
X_test = test_data[features]
y_test = test_data[target]

# Predict using the trained model
y_pred = rf_classifier.predict(X_test)

# Evaluate and print model performance on test data
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the IMAGE model on new test data:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

time.sleep(100000)

 # Path to the new test file
#######predictions = load_and_predict(test_file_path)
#print(predictions)

time.sleep(100000)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer



import pandas as pd

# Load Train.csv into a pandas DataFrame
train_df = pd.read_csv('train.csv')

# Dictionary to map the current column names in Train.csv to the new names from Week1.csv
column_mapping = {
    'Yards': 'yardsToGo',
    'Quarter': 'quarter',
    'Down': 'down',
    'YardLine': 'yardlineNumber',
    'HomeScoreBeforePlay': 'preSnapHomeScore',
    'VisitorScoreBeforePlay': 'preSnapVisitorScore',
    'PossessionTeam': 'possessionTeam',
    'DefendersInTheBox': 'defendersInTheBox',
    'GameClock': 'gameClock',
}

# Rename the columns
train_df.rename(columns=column_mapping, inplace=True)

# Optionally, save the renamed DataFrame back to a CSV if needed
train_df.to_csv('Train_Renamed.csv', index=False)

# Display the DataFrame to verify the changes
print(train_df.head())

time.sleep(10000)
















# Function to load and preprocess data
def load_and_preprocess_data(week_numbers):
    dfs = []
    for i in week_numbers:
        file_path = f"offense_only_final_data_week{i}_2024.csv"
        df_week = pd.read_csv(file_path)
        
        
        df_week['centroid_x'] = df_week.groupby(['gameId', 'playId'])['x'].transform('mean')
        df_week['centroid_y'] = df_week.groupby(['gameId', 'playId'])['y'].transform('mean')
        df_week['spread_x'] = df_week.groupby(['gameId', 'playId'])['x'].transform('std')
        df_week['spread_y'] = df_week.groupby(['gameId', 'playId'])['y'].transform('std')
        df_week['max_speed'] = df_week.groupby(['gameId', 'playId'])['s'].transform('max')
        df_week['max_acceleration'] = df_week.groupby(['gameId', 'playId'])['a'].transform('max')
        
        # Keep the first occurrence to avoid duplicate plays in the dataset
        df_week = df_week.drop_duplicates(subset=['gameId', 'playId'])
        dfs.append(df_week)
        
    
        
    # Combine all weeks into a single DataFrame
    df_combined = pd.concat(dfs, ignore_index=True)
    return df_combined

# Load and preprocess training and testing data
train_weeks = range(1, 5)
test_weeks = range(5, 8)

df_train = load_and_preprocess_data(train_weeks)
df_test = load_and_preprocess_data(test_weeks)

df_train = df_train.dropna(subset=['offensive_formation'])
df_test = df_test.dropna(subset=['offensive_formation'])

# Correcting the aggregation of dummy-encoded player positions
player_positions_train = pd.get_dummies(df_train.set_index(['gameId', 'playId'])['player_position']).groupby(['gameId', 'playId']).agg('max').reset_index()
player_positions_test = pd.get_dummies(df_test.set_index(['gameId', 'playId'])['player_position']).groupby(['gameId', 'playId']).agg('max').reset_index()

# Ensure the subsequent merging operations are adjusted to work with the corrected aggregation
# Since 'gameId' and 'playId' are now part of the index after groupby, ensure they are correctly reset
# (The provided correction already handles this with .reset_index())

# The rest of your code for merging the player positions back into your datasets, preparing features and targets, and fitting the model remains unchanged.

df_train = pd.merge(df_train.drop_duplicates(subset=['gameId', 'playId']), player_positions_train, on=['gameId', 'playId'])
df_test = pd.merge(df_test.drop_duplicates(subset=['gameId', 'playId']), player_positions_test, on=['gameId', 'playId'])


# Then, prepare features and target variables as before
y_train = df_train['offensive_formation']
y_test = df_test['offensive_formation']
X_train = df_train.drop(['offensive_formation', 'gameId', 'playId', 'displayName', 'jerseyNumber', 'x', 'y', 's', 'a', 'player_position'], axis=1)
X_test = df_test.drop(['offensive_formation', 'gameId', 'playId', 'displayName', 'jerseyNumber', 'x', 'y', 's', 'a', 'player_position'], axis=1)



# Encode categorical variables and scale numerical features
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
numerical_features = X_train.select_dtypes(exclude=['object']).columns.tolist()

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Impute NaNs
    ('scaler', StandardScaler())])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# Define the model pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

# Fit the model on the training data
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy on Weeks 5-7 Data: {accuracy}')





#############OFFENSE FORMATION MODEL
















import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
data = pd.read_csv('NEW_FINAL_SORTED_PLAYS_NEW.csv')

# Remove any rows with missing values
data.dropna(inplace=True)

# Preprocess 'gameClock' - Convert 'gameClock' from 'mm:ss' to total seconds
def convert_time_to_seconds(time_str):
    minutes, seconds = map(int, time_str.split(':'))
    return minutes * 60 + seconds

data['gameClock'] = data['gameClock'].apply(convert_time_to_seconds)

# Encode categorical variables
label_encoders = {}
categorical_columns = ['possessionTeam', 'defensiveTeam', 'yardlineSide', 'offenseFormation', 'playDirection']
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Features and target variable
features = ['quarter', 'down', 'yardsToGo', 'possessionTeam', 'defensiveTeam', 'yardlineSide',
            'yardlineNumber', 'gameClock', 'preSnapHomeScore', 'preSnapVisitorScore',
            'defendersInTheBox', 'offenseFormation', 'playDirection']
target = 'PlayType'

X = data[features]
y = data[target]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the model:", accuracy)

# Display the classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Optionally, display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)


time.sleep(100000)



import pandas as pd
"""
# Load the CSV file
""" df = pd.read_csv('sorted_plays_with_playtype.csv')

# Define the columns to keep
columns_to_keep = [
    "gameId", "playId", 'quarter', 'down', 'yardsToGo', 'possessionTeam', 'defensiveTeam', 
    'yardlineSide', 'yardlineNumber', 'gameClock', 'preSnapHomeScore', 
    'preSnapVisitorScore', 'defendersInTheBox', 'offenseFormation', 
    'PlayType', 'passProbability', "playDescription"
]

# Filter the dataframe to keep only the specified columns
filtered_df = df[columns_to_keep]

# Save the filtered dataframe to a new CSV file
filtered_df.to_csv('final_data_plays_2022.csv', index=False)

print("Filtered DataFrame saved as filtered_plays.csv")

 """


import pandas as pd

# Initialize an empty DataFrame to hold all the data
combined_df = pd.DataFrame()

""" # Loop through the numbers 1 to 7 to load each file
for i in range(1, 10):
    # Construct the file name
    file_name = f"NEW_tracking_week_{i}.csv"
    
    # Load the current CSV file
    current_df = pd.read_csv(file_name)
    
    # Append the current DataFrame to the combined DataFrame
    combined_df = pd.concat([combined_df, current_df], ignore_index=True)

# Optionally, save the combined DataFrame to a new CSV file
combined_df.to_csv('NEW_TRACKER.csv', index=False)

print("All data has been combined and saved as combined_final_data_2024.csv")

 """
""" import pandas as pd

# Load the previously merged dataset
merged_data = pd.read_csv('NEW_TRACKER.csv')

# Filter out rows where playDirection is missing
cleaned_data = merged_data.dropna(subset=['playDirection'])

# Save the cleaned DataFrame to a new CSV file
cleaned_data.to_csv('NEW_FINAL.csv', index=False)

print("Rows with missing playDirection have been removed and the file is saved as cleaned_final_data_plays_2022.csv")
 
time.sleep(10000)
 """

import pandas as pd

# Load the datasets
offense_data = pd.read_csv('NEW_TRACKER.csv')
play_data = pd.read_csv('NEW_PLAYS.csv')

# Since playDirection is the same for all entries with the same gameId and playId,
# we can drop duplicates after selecting these columns
offense_data_reduced = offense_data[['gameId', 'playId', 'playDirection']].drop_duplicates()

# Merge the playDirection into the plays data based on gameId and playId
merged_data = play_data.merge(offense_data_reduced, on=['gameId', 'playId'], how='left')

# Save the merged DataFrame to a new CSV file
merged_data.to_csv('NEW_TRACKER_PLAYS.csv', index=False)

print("The playDirection has been merged and the updated file is saved as updated_final_data_plays_2022.csv")







time.sleep(1000000)
final_data = pd.read_csv("final_data_week7_2024.csv")
plays_data = pd.read_csv("plays2024.csv")


# Re-merging the dataframes to label 'Offense' or 'Defense' accurately based on 'possessionTeam' and 'defensiveTeam'
merged_data_revised = pd.merge(final_data, plays_data[['gameId', 'playId', 'possessionTeam', 'defensiveTeam']], on=['gameId', 'playId'], how='left')

# Creating a new column 'teamType' again to indicate whether the row corresponds to 'Offense' or 'Defense'
merged_data_revised['teamType'] = merged_data_revised.apply(lambda row: 'Offense' if row['club'] == row['possessionTeam'] else ('Defense' if row['club'] == row['defensiveTeam'] else 'Unknown'), axis=1)

# Selecting only the specified columns to keep in the final dataframe
final_columns = ['gameId', 'playId', 'nflId', 'displayName', 'jerseyNumber', 'club', 'playDirection', 'x', 'y', 's', 'a', 'dis', 'o', 'dir', 'player_position', 'offensive_formation', 'teamType']
modified_data_revised = merged_data_revised[final_columns]

# Saving this refined dataframe to a new CSV file
refined_file_path = 'final_week7_data_test_2024.csv'
modified_data_revised.to_csv(refined_file_path, index=False)

refined_file_path

time.sleep(100000) 
""" 
import pandas as pd

# import pandas as pd

# Load the CSV file
df = pd.read_csv('sorted_plays_with_playtype.csv')

# Group by 'offenseFormation' and 'PlayType' and count occurrences
play_counts = df.groupby(['offenseFormation', 'PlayType']).size().unstack(fill_value=0)

# The unstack operation will create a DataFrame with each row as an 'offenseFormation' and each column as a 'PlayType'
# Columns for 'Pass' and 'Run' will show the count of each type of play for each formation
# Fill any missing values with 0 (using fill_value in unstack), which is useful if some formations have no passes or runs

# Reset the index so 'offenseFormation' is a column and not an index
play_counts = play_counts.reset_index()

# Print the results to check
print(play_counts)

# Optionally, save the results to a new CSV file
play_counts.to_csv('142offense_formation_play_counts.csv', index=False)
time.sleep(10000)
 """


""" 

file_path_new = 'plays2024.csv'


# Let's attempt to process the plays2024.csv file again, this time dropping all columns except the specified ones
# and then we'll display the first few rows to ensure the operation was successful.
df_plays = pd.read_csv("games.csv")
df_plays = df_plays.sort_values(by=['gameId'])
df_plays.to_csv("sorted_games_2024.csv", index=False) """




""" 



try:
    # Load the CSV file again, in case the DataFrame was modified previously
    df_plays = pd.read_csv(file_path_new)

    # Specify the columns to retain
    columns_to_keep = [
        'gameId', 'playId', 'playDescription', 'quarter', 'down', 'yardsToGo', 'possessionTeam',
        'defensiveTeam', 'yardlineSide', 'yardlineNumber', 'gameClock', 'preSnapHomeScore', 'preSnapVisitorScore', "offenseFormation"
    ]

    # Drop all other columns except the ones specified
    df_filtered = df_plays[columns_to_keep]
    df_filtered = df_filtered.sort_values(by=['gameId', 'playId'])

    # Display the first few rows to ensure correctness

except Exception as e:
    print(f"An error occurred: {e}")


df_filtered.to_csv("sorted_plays_2024.csv", index = False) """


""" 
# Load the CSV file
df = pd.read_csv('plays2024.csv')

# Function to determine play type
def determine_play_type(row):
    if pd.notna(row['passResult']) and row['passLength'] != 'NA':
        return 'Pass'
    else:
        return 'Run'

# Apply the function to create a new column
df['PlayType'] = df.apply(determine_play_type, axis=1)

# Display the first few rows of the DataFrame to verify the new column
print(df.head())

# Save the DataFrame with the new column to a new CSV file if needed
df.to_csv('sorted_plays_with_playtype.csv', index=False)
time.sleep(10000)

 """
""" 
# Load the CSV file
df = pd.read_csv('sorted_plays_2024.csv')

# Find the first two unique game IDs
first_two_game_ids = df['gameId'].unique()[:5]

# Filter the DataFrame to only include rows with the first two game IDs
filtered_df = df[df['gameId'].isin(first_two_game_ids)]

# Check unique formations and count occurrences in the filtered DataFrame
formation_counts = filtered_df['offenseFormation'].value_counts()

# Print the unique formations and their counts
print(formation_counts)
time.sleep(100000) """

""" 
# Load the CSV file
df = pd.read_csv('final_train.csv')

# Check unique formations and count occurrences
formation_counts = df['OffenseFormation'].value_counts()

# Print the unique formations and their counts
print(formation_counts)

time.sleep(10000) """





"""

try:
    # Load the CSV file
    df_plays = pd.read_csv(file_path_new)

    # Group by 'gameId' and then sort by 'playId' within each group
    df_sorted = df_plays.sort_values(by=['gameId', 'playId'])

    # Display the first few rows of the sorted DataFrame to verify the operation
except Exception as e:
    print(f"An error occurred: {e}")

df_sorted.to_csv("sorted_plays_2024.csv", index = False)
time.sleep(1000)
csv_data_2020 = "train.csv"
output_path_2020 = "final_tracking_data_2020.csv"
df = pd.read_csv(csv_data_2020)


# Rename columns to match your desired names, if they exist in your DataFrame
rename_columns = {
    'GameId': 'gameId',
    'PlayId': 'playId',
    # Add more mappings here based on the actual column names in your DataFrame
    'Team': 'team',
    'X': 'x',
    'Y': 'y',
    'S': 's',
    'A': 'a',
    'Dis': 'dis',
    'Orientation': 'o',
    'Dir': 'dir',
    'Position': 'player_position',  # Assuming 'Position' is the column to be renamed to 'player_position'
    'OffenseFormation': 'offensive_formation'  # Assuming this is the correct column to map
}
df.rename(columns=rename_columns, inplace=True)

# Filter the DataFrame to keep only the specified columns
filtered_columns = [
    'gameId', 'playId', 'displayName', 'jerseyNumber', 'team',
    'playDirection', 'x', 'y', 's', 'a', 'dis', 'o', 'dir',
    'player_position', 'offensive_formation'
]
df_filtered = df[filtered_columns]

# Save the filtered DataFrame to a new CSV file
df_filtered.to_csv('2020datafinal.csv', index=False)
time.sleep(10000)"""






import time

time.sleep(1000)


#print(nfl_id)



""" filtered_rows = play_info_df[(play_info_df['playId'] == 384) & (play_info_df['gameId'] == 2022091800)]

# Print the filtered rows
print(filtered_rows)
import time

# Assuming filtered_rows is your DataFrame from the previous step
if not filtered_rows.empty:
    # Access the 'offenseFormation' of the first (and presumably only) row
    offenseFormation = filtered_rows["offenseFormation"].iloc[0]
    print(str(offenseFormation))

else:
    print("No rows match the filter criteria.")


# Add a new column for player position in the tracking data DataFrame, initialize with None
tracking_data['player_position'] = None
tracking_data['offensive_formation'] = None
# Create a dictionary from the players DataFrame mapping nflId to position
nflid_to_position = player_info.set_index('nflId')['position'].to_dict()

# Update the player_position column based on nflId using the map function
tracking_data['player_position'] = tracking_data['nflId'].map(nflid_to_position)

# Check the first few rows to verify the new column
#tracking_data.drop('time', axis=1, inplace=True)
#tracking_data.drop('frameId', axis=1, inplace=True)
# Check the first few rows to verify the column has been removed
print(tracking_data)



# Prepare to enrich tracking data with offenseFormation information
for play_id, group in grouped:
    # For each group, extract the unique gameId (assuming all rows in a group share the same gameId)
    unique_game_id = group['gameId'].iloc[0]

    # Find the matching offenseFormation in the play_info_df
    matching_play_info = play_info_df[(play_info_df['playId'] == play_id) & (play_info_df['gameId'] == unique_game_id)]
    
    # Check if matching_play_info is not empty
    if not matching_play_info.empty:
        offenseFormation = matching_play_info['offenseFormation'].iloc[0]
        
        # Print the offenseFormation for verification
        print(f"PlayId: {play_id}, GameId: {unique_game_id}, OffenseFormation: {offenseFormation}")
        
        # If you want to add/update the offenseFormation to your group (or to a DataFrame),
        # you can do so. Here's an example of updating the original DataFrame:
        # Note: This operation should ideally be done outside the loop for efficiency,
        # and using a more optimized approach like merging.
        tracking_data.loc[(tracking_data['playId'] == play_id) & (tracking_data['gameId'] == unique_game_id), 'offensive_formation'] = offenseFormation
    else:
        print(f"No matching play found for PlayId: {play_id} and GameId: {unique_game_id}")

    
print(tracking_data)
# Save the filtered DataFrame to a new CSV file
tracking_data.to_csv("final_data_week1_2024.csv", index=False)
import time
time.sleep(1000)
"""
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load your dataset
df = pd.read_csv('final_data_week1_2024.csv')

df.drop('s', axis=1, inplace=True)
df.drop('a', axis=1, inplace=True)
# Drop rows where the target variable is NaN
df.to_csv("final_data_week1_2024.csv", index=False)
print(df) 



df = pd.read_csv('final_data_week1_2024.csv')

# Identify the rows for the football
football_data = df[df['displayName'] == 'football']

# Iterate through each unique playId to adjust player positions
for play_id in df['playId'].unique():
    # Get the football position for this play
    football_pos = football_data[football_data['playId'] == play_id][['x', 'y']].iloc[0]
    
    # Find the indices for this play (excluding the football)
    play_indices = df[(df['playId'] == play_id) & (df['displayName'] != 'football')].index
    
    # Adjust positions
    df.loc[play_indices, 'x'] -= football_pos['x']
    df.loc[play_indices, 'y'] -= football_pos['y']
df.to_csv("final_data_week1_2024.csv", index=False)

df.dropna(subset=['offensive_formation'], inplace=True)

# Proceed with splitting the data, feature preprocessing, and model training as before

# Separate features and target variable
X = df[['club', 'playDirection', 'x', 'y', 'dis', 'o', 'dir', 'event', 'player_position']]
y = df['offensive_formation']

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Preprocessing steps
numeric_features = ['x', 'y', 'dis', 'o', 'dir']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['club', 'playDirection', 'event', 'player_position']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

    



# Define the model pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', LogisticRegression(max_iter=1000, random_state=42))])

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}') """
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
""" offense_only = pd.read_csv("final_week7_data_test_2024.csv")
# Dropping all rows where 'teamType' equals 'Defense'
offense_only_data = offense_only[offense_only['teamType'] == 'Offense']

# Saving this offense-only dataframe to a new CSV file
offense_only_file_path = 'offense_only_final_data_week7_2024.csv'
offense_only_data.to_csv(offense_only_file_path, index=False)

offense_only_file_path
time.sleep(10000) """

""" 
# Initialize list to store accuracy scores for each file
accuracy_scores = []

# Loop through all 7 files
for i in range(1, 8):
    file_name = f'offense_only_final_data_week{i}_2024.csv'
    df = pd.read_csv(file_name)
    df.drop('displayName', axis=1, inplace=True)
    df.drop('jerseyNumber', axis=1, inplace=True)
    # Drop rows where the target variable is NaN
    df.dropna(subset=['offensive_formation'], inplace=True)

    # Define features and target
    X = df.drop(['offensive_formation'], axis=1)  # Adjust as needed
    y = df['offensive_formation']

    # Define categorical and numerical features
    categorical_features = ['club', 'playDirection', 'player_position']
    numerical_features = ['x', 'y', 'dis', 'o', 'dir', "s", "a"]  # Adjust based on your data

    # Create transformers for preprocessing
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))])

    # Combine transformers into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])

    # Create the model pipeline
    model = make_pipeline(preprocessor, RandomForestClassifier(n_estimators=100, random_state=42))

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

    # Print individual file accuracy
    print(f'Accuracy for {file_name}: {accuracy}')
    #time.sleep(1000000)

# Calculate the average accuracy across all files
final_accuracy = sum(accuracy_scores) / len(accuracy_scores)
print(f'Final Average Accuracy: {final_accuracy}')
 """



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

""" # Combine data from weeks 1-3 for training
train_dfs = []
for i in range(1, 5):
    file_name = f'offense_only_final_data_week{i}_2024.csv'
    df = pd.read_csv(file_name)
    df.drop(['displayName', 'jerseyNumber'], axis=1, inplace=True)
    df.dropna(subset=['offensive_formation'], inplace=True)
    train_dfs.append(df)
train_df = pd.concat(train_dfs, ignore_index=True)

# Combine data from weeks 4-7 for testing
test_dfs = []
for i in range(5, 8):
    file_name = f'offense_only_final_data_week{i}_2024.csv'
    df = pd.read_csv(file_name)
    df.drop(['displayName', 'jerseyNumber'], axis=1, inplace=True)
    df.dropna(subset=['offensive_formation'], inplace=True)
    test_dfs.append(df)
test_df = pd.concat(test_dfs, ignore_index=True) """



#TRAIN TEST SPLIT FUNC ABOVE
""" 
# Define features and target for training and testing
X_train = train_df.drop(['offensive_formation'], axis=1)
y_train = train_df['offensive_formation']
X_test = test_df.drop(['offensive_formation'], axis=1)
y_test = test_df['offensive_formation']

# Define categorical and numerical features
categorical_features = ['club', 'playDirection', 'player_position']
numerical_features = ['x', 'y', 'dis', 'o', 'dir', "s", "a"]

# Create transformers for preprocessing
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))])

# Combine transformers into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# Create the model pipeline
model = make_pipeline(preprocessor, RandomForestClassifier(n_estimators=100, random_state=42))

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Final Accuracy on Weeks 4-7: {accuracy}')
 """


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

""" def aggregate_play_data(df):
    # Aggregating numerical features
    num_aggregations = {
        'x': ['mean', 'max', 'min'],
        'y': ['mean', 'max', 'min'],
        's': ['max'],
        'a': ['mean', 'max'],
        'dis': ['sum'],
        'o': ['mean'],
        'dir': ['mean']
    }
    num_agg = df.groupby(['gameId', 'playId'], as_index=False).agg(num_aggregations)
    num_agg.columns = ['_'.join(col).strip() if col[1] else col[0] for col in num_agg.columns.values]
    
    # Handling categorical data more robustly
    cat_agg = df.groupby(['gameId', 'playId'])['player_position'].agg(lambda x: x.value_counts().index[0] if not x.empty else np.nan).reset_index()
    cat_agg.rename(columns={'player_position': 'player_position_mode'}, inplace=True)
    
    # Merging aggregated numerical and categorical data
    agg_df = pd.merge(num_agg, cat_agg, on=['gameId', 'playId'])

    # Including 'offensive_formation' in the aggregated dataframe
    agg_df = pd.merge(agg_df, df[['gameId', 'playId', 'offensive_formation']].drop_duplicates(), on=['gameId', 'playId'], how='left')
    
    return agg_df """


""" def aggregate_play_data(df):
    # First, determine quartiles for each play. This requires a unique identifier for each row in the original dataset.
    df['quartile'] = df.groupby('playId')['frameId'].transform(lambda x: pd.qcut(x, 4, labels=False) + 1)
    
    # Calculate mean 'x' and 'y' for each quartile of each play
    quartile_means = df.groupby(['gameId', 'playId', 'quartile'])['x', 'y'].mean().unstack(level=-1)
    quartile_means.columns = [f'{col[0]}_q{col[1]}_mean' for col in quartile_means.columns]
    
    # Calculate max for 's' and 'a'
    max_sa = df.groupby(['gameId', 'playId'])['s', 'a'].max().reset_index()
    max_sa.columns = [f'{col}_max' for col in max_sa.columns if col not in ['gameId', 'playId']] + ['gameId', 'playId']
    
    # Handling categorical data for 'player_position'
    cat_agg = df.groupby(['gameId', 'playId'])['player_position'].agg(lambda x: x.value_counts().index[0] if not x.empty else np.nan).reset_index()
    cat_agg.rename(columns={'player_position': 'player_position_mode'}, inplace=True)
    
    # Merge the aggregated data
    agg_df = pd.merge(quartile_means.reset_index(), max_sa, on=['gameId', 'playId'])
    agg_df = pd.merge(agg_df, cat_agg, on=['gameId', 'playId'])

    # Including 'offensive_formation' in the aggregated dataframe
    agg_df = pd.merge(agg_df, df[['gameId', 'playId', 'offensive_formation']].drop_duplicates(), on=['gameId', 'playId'], how='left')
    
    return agg_df """
""" 
def aggregate_play_data(df):
    # Calculate overall mean, max, min for 'x' and 'y'
    overall_agg = df.groupby(['gameId', 'playId'])['x', 'y'].agg(['mean', 'max', 'min']).reset_index()
    overall_agg.columns = ['_'.join(col).strip() if col[1] else col[0] for col in overall_agg.columns.values]
    
    # Calculate max for 's' and 'a'
    max_sa = df.groupby(['gameId', 'playId'])[['s', 'a']].max().reset_index()
    max_sa.columns = ['_'.join([col, 'max']) if col in ['s', 'a'] else col for col in max_sa.columns]
    
    # Merge the aggregated numerical data
    agg_df = pd.merge(overall_agg, max_sa, on=['gameId', 'playId'])
    
    # Assuming 'player_position' and 'offensive_formation' are directly from df, adjust if needed
    # No aggregation for 'player_position' as each row now corresponds to a play, not individual players
    # Directly include 'offensive_formation'
    agg_df = pd.merge(agg_df, df[['gameId', 'playId', 'player_position', 'offensive_formation']].drop_duplicates(), on=['gameId', 'playId'], how='left')
    
    return agg_df

# Continue with loading data, creating training and testing sets, and training/testing the model as before





# Load your data here and split it into training and testing sets as per the provided instructions
# Assume train_df and test_df are already loaded and preprocessed

# Aggregate the training and testing data
train_df_agg = aggregate_play_data(train_df)
test_df_agg = aggregate_play_data(test_df)

# Define features and target for aggregated data
X_train_agg = train_df_agg.drop(['gameId', 'playId', 'offensive_formation'], axis=1)
y_train_agg = train_df_agg['offensive_formation']
X_test_agg = test_df_agg.drop(['gameId', 'playId', 'offensive_formation'], axis=1)
y_test_agg = test_df_agg['offensive_formation']

# Adjusting the preprocessing pipelines for the aggregated features
categorical_features = ['player_position_mode']
numerical_features = [col for col in X_train_agg.columns if col not in categorical_features]

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# Creating the model pipeline
model = make_pipeline(preprocessor, RandomForestClassifier(n_estimators=100, random_state=42))

# Training the model on aggregated training data
model.fit(X_train_agg, y_train_agg)

# Making predictions on the aggregated test data
y_pred_agg = model.predict(X_test_agg)

# Evaluating the model
accuracy = accuracy_score(y_test_agg, y_pred_agg)
print(f'Final Accuracy on Aggregated Weeks 4-7 Data: {accuracy}') """

# Retry loading the dataset due to the previous connection issue
import pandas as pd

# Load the dataset
""" df = pd.read_csv('train.csv')

# Display the first few rows of the dataframe to understand its structure
df.head() """



import pandas as pd

# Load your dataset
""" df = pd.read_csv('train.csv')

# Assuming you have a way to determine the home and away teams for each game
# For example, if you have columns 'HomeTeamAbbr' and 'VisitorTeamAbbr', and a column 'club' indicating home or away
# You can create a new column for the actual team name

# Function to map the team abbreviation
def map_team_abbr(row):
    if row['Team'] == 'home':
        return row['HomeTeamAbbr']
    elif row['Team'] == 'away':
        return row['VisitorTeamAbbr']
    else:
        return None

# Apply the function to create a new column with the actual team abbreviation
df['ActualTeamAbbr'] = df.apply(map_team_abbr, axis=1)

# Save the adjusted dataframe back to a csv if needed
df.to_csv('final_train.csv', index=False) """

""" 
import pandas as pd

# Assuming df is your DataFrame after loading your dataset and adjusting team names
# Load your dataset (this line is just a placeholder; replace it with your actual data loading code)
df = pd.read_csv('final_train.csv')

# Filter the DataFrame to only include rows where the ActualTeamAbbr matches the HomeTeamAbbr, indicating offense players
offense_df = df[df['ActualTeamAbbr'] == df['HomeTeamAbbr']]

# Now, offense_df contains only the data for offensive players for each play and game ID combination
# You can save this subset to a new file or proceed with further analysis
offense_df.to_csv('offense_test_final.csv', index=False) """




#####OFFENSE FORMATION MODEL

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer



import pandas as pd

# Load Train.csv into a pandas DataFrame
""" train_df = pd.read_csv('train.csv')

# Dictionary to map the current column names in Train.csv to the new names from Week1.csv
column_mapping = {
    'Yards': 'yardsToGo',
    'Quarter': 'quarter',
    'Down': 'down',
    'YardLine': 'yardlineNumber',
    'HomeScoreBeforePlay': 'preSnapHomeScore',
    'VisitorScoreBeforePlay': 'preSnapVisitorScore',
    'PossessionTeam': 'possessionTeam',
    'DefendersInTheBox': 'defendersInTheBox',
    'GameClock': 'gameClock',
}

# Rename the columns
train_df.rename(columns=column_mapping, inplace=True)

# Optionally, save the renamed DataFrame back to a CSV if needed
train_df.to_csv('Train_Renamed.csv', index=False)

# Display the DataFrame to verify the changes
print(train_df.head())

time.sleep(10000) """


""" # Function to load and preprocess data
def load_and_preprocess_data(week_numbers):
    dfs = []
    for i in week_numbers:
        file_path = f"offense_only_final_data_week{i}_2024.csv"
        df_week = pd.read_csv(file_path)
        
        
        df_week['centroid_x'] = df_week.groupby(['gameId', 'playId'])['x'].transform('mean')
        df_week['centroid_y'] = df_week.groupby(['gameId', 'playId'])['y'].transform('mean')
        df_week['spread_x'] = df_week.groupby(['gameId', 'playId'])['x'].transform('std')
        df_week['spread_y'] = df_week.groupby(['gameId', 'playId'])['y'].transform('std')
        df_week['max_speed'] = df_week.groupby(['gameId', 'playId'])['s'].transform('max')
        df_week['max_acceleration'] = df_week.groupby(['gameId', 'playId'])['a'].transform('max')
        
        # Keep the first occurrence to avoid duplicate plays in the dataset
        df_week = df_week.drop_duplicates(subset=['gameId', 'playId'])
        dfs.append(df_week)
        
    
        
    # Combine all weeks into a single DataFrame
    df_combined = pd.concat(dfs, ignore_index=True)
    return df_combined

# Load and preprocess training and testing data
train_weeks = range(1, 5)
test_weeks = range(5, 8)

df_train = load_and_preprocess_data(train_weeks)
df_test = load_and_preprocess_data(test_weeks)

df_train = df_train.dropna(subset=['offensive_formation'])
df_test = df_test.dropna(subset=['offensive_formation'])

# Correcting the aggregation of dummy-encoded player positions
player_positions_train = pd.get_dummies(df_train.set_index(['gameId', 'playId'])['player_position']).groupby(['gameId', 'playId']).agg('max').reset_index()
player_positions_test = pd.get_dummies(df_test.set_index(['gameId', 'playId'])['player_position']).groupby(['gameId', 'playId']).agg('max').reset_index()

# Ensure the subsequent merging operations are adjusted to work with the corrected aggregation
# Since 'gameId' and 'playId' are now part of the index after groupby, ensure they are correctly reset
# (The provided correction already handles this with .reset_index())

# The rest of your code for merging the player positions back into your datasets, preparing features and targets, and fitting the model remains unchanged.

df_train = pd.merge(df_train.drop_duplicates(subset=['gameId', 'playId']), player_positions_train, on=['gameId', 'playId'])
df_test = pd.merge(df_test.drop_duplicates(subset=['gameId', 'playId']), player_positions_test, on=['gameId', 'playId'])


# Then, prepare features and target variables as before
y_train = df_train['offensive_formation']
y_test = df_test['offensive_formation']
X_train = df_train.drop(['offensive_formation', 'gameId', 'playId', 'displayName', 'jerseyNumber', 'x', 'y', 's', 'a', 'player_position'], axis=1)
X_test = df_test.drop(['offensive_formation', 'gameId', 'playId', 'displayName', 'jerseyNumber', 'x', 'y', 's', 'a', 'player_position'], axis=1)



# Encode categorical variables and scale numerical features
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
numerical_features = X_train.select_dtypes(exclude=['object']).columns.tolist()

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Impute NaNs
    ('scaler', StandardScaler())])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# Define the model pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

# Fit the model on the training data
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy on Weeks 5-7 Data: {accuracy}')
 """




#############OFFENSE FORMATION MODEL








#######OFFENSIVE MODEL PREDICTOR FOR TRACKING DATA ####
##### """


""" print(f'Model Accuracy on Weeks 5-7 Data: {accuracy_score(y_test, y_pred)}')
print(f'Predictions saved to "predicted_offensive_formations.csv".')
import time
time.sleep(100000)
 """

""" import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Path to your dataset
dataset_path = '/path/to/your/image/dataset'  # Update this path

# Initialize the base pre-trained model
base_model = ResNet50(weights='imagenet', include_top=False)

# Add new layers for our specific task
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(number_of_formations, activation='softmax')(x)  # number_of_formations should be set to the number of unique formations

# Define the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model to not train them again
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Set up data augmentation and generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # assuming 20% of the data is used for validation
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32
)

# Save model
model.save('nfl_formation_classifier.h5')
 """



