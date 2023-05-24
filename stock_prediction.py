
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam, RMSprop
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import pandas_datareader.data as web
import pandasgui as pg
from sklearn import preprocessing
from collections import deque
import random


df = pd.read_csv("C:/Users/....../stock market project/OXY.csv", names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

SEQ_LEN = 60  # how long of a preceeding sequence to collect for RNN
FUTURE_PERIOD_PREDICT = 2  # כמה ימים אנו רוצים לצפות עבור המנייה
RATIO_TO_PREDICT = "OVV"
EPOCHS = 10      #  מספר הפעמים שהמודל ירוץ על הדאטה
BATCH_SIZE = 16  #  יצירת קבוצות שבגודל הזה יתחלק לתת קבוצה הדאטה הגדול וירוץ על התתי קבוצות המודל
NAME = 'second project'

main_df = pd.DataFrame() # יצירת דאטה ריקה


def classify(current, future):   # בודק האם בתקופה הבאה המחיר יעלה או ירד
    if float(future) > float(current):
        return 1
    else:
        return 0

def preprocess_df(df):
    df = df.drop(labels='Future', axis=1)   # מחיקת עמודת הפיוצר
    df = df.drop(labels='Date', axis=1)     # מחיקת עמודת התאריך (על מנת שיהיה רק ערכים מספריים)
    for col in df.columns:
        if col not in ["Target", 'Date']:
            df[col] = df[col].pct_change()  #  מחסר בין שני תאים באותה עמודה ומחלק בראשון - זאת אומרת אחוז השינוי - מנרמל את היחסים בין המשתנים לבין 0 ו1
            df[col] = preprocessing.scale(df[col].values)   # מעדכן את הנתונים כך שהם יהיו עם  ממוצע 0 וסטיית תקן של 1
    df.dropna(inplace=True)

    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)  # ככל שמכניסים אליה ערכים היא שומרת עד אורך שהזנתי, ואז בליפו מוציאה את האחרון ישר כאשר מכניסים לה איבר נוסף (תןר)

    for i in df.values:   # כל i הוא שורה בטבלה
        prev_days.append([n for n in i[:-1]])  # כאן מוסיפים רשימה ללא הטארגט (שזה הניבוי אם עולה או יורד)
        if len(prev_days) == SEQ_LEN: #  כאשר התור מתמלא (ול60 ערכים הראשונים אין להם עדיין תוצאת טארגט, כי רק אחרי שמתמלאים 60 ערכים אז זה מתחיל להיכנס)
            sequential_data.append([np.array(prev_days), i[-1]])  #נוסיף לדאטה המחולק רשימה של הדאטה ללא הטארגט, ואת הרשימה של הניבוי (שיודעים מה הוא כי זה נתונים מההסטוריה)

    random.shuffle(sequential_data)

# Making Balance for the data  - כדי שיהיו בערך אותם כמות של עלייה וירידה (מכירה וקנייה של המנייה)
    buys = []
    sells = []

    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])

    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells))

    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys + sells
    random.shuffle(sequential_data)


    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), y

def creating_test(df):
    df = df.drop(labels='Future', axis=1)   # מחיקת עמודת הפיוצר
    df = df.drop(labels='Date', axis=1)     # מחיקת עמודת התאריך (על מנת שיהיה רק ערכים מספריים)
    for col in df.columns:
        if col not in ["Target", 'Date']:
            df[col] = df[col].pct_change()  # מחסר בין שני תאים באותה עמודה ומחלק בראשון - זאת אומרת אחוז השינוי
            df[col] = preprocessing.scale(df[col].values)   # מעדכן את הנתונים כך שהם יהיו עם  ממוצע 0 וסטיית תקן של 1
    df.dropna(inplace=True)

    prev_days = deque(maxlen=SEQ_LEN)  # ככל שמכניסים אליה ערכים היא שומרת עד אורך שהזנתי, ואז בליפו מוציאה את האחרון ישר כאשר מכניסים לה איבר נוסף (תןר)

    for i in df.values:   # כל i הוא שורה בטבלה
        prev_days.append([n for n in i[:-1]])  # כאן מוסיפים רשימה ללא הטארגט (שזה הניבוי אם עולה או יורד)
    return np.array([np.array(prev_days)])



ratios = ["OXY", "OVV", "XLE"]  # המניות שאני מוסיף לניתוח
for ratio in ratios:  # begin iteration
    print(ratio)
    dataset = f"C:/Users/......./stock market project/{ratio}.csv"  # get the full path to the file.
    df = pd.read_csv(dataset, names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])  # read in specific file

    # rename volume and close to include the ticker so we can still which close/volume is which:
    df.rename(columns={"Close": f"{ratio}_close", "Volume": f"{ratio}_volume"}, inplace=True)

    df[f"{ratio}_close"] = df[f"{ratio}_close"].astype('float')
    df[f"{ratio}_volume"] = df[f"{ratio}_volume"].astype('float')
    df.set_index("Date", inplace=True)  # set time as index so we can join them on this shared time
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]  # ignore the other columns besides price and volume

    if len(main_df)==0:  # if the dataframe is empty
        main_df = df  # then it's just the current df
    else:  # otherwise, join this data to the main one
        main_df = main_df.join(df)


main_df['Future'] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)   # להוסיף עמודה של הזזה של המחיר בכמות ימים שנרצה
main_df['Target'] = list(map(classify, main_df[f"{RATIO_TO_PREDICT}_close"], main_df['Future']))  # הוספת עמודה של האם המחיר יהיה בעוד כמה ימים מעל או מתחת למחיר של היום

new_df = main_df.reset_index(drop=False)  #  יצירת טבלת נתונים חדשה שבא הוספתי אינדקסים לשורות שמסודרות לפי התאריך

print(new_df.shape)

times = sorted(new_df.index.values)  # הכנסת כל ערכי האינדקס מימויינם והכנסה לרשימה
last_5pct = sorted(new_df.index.values)[-int(0.05*len(times))]  # מציאת האינדקס ה95% כדי שיהיה לאימון

validation_main_df = new_df[(new_df.index >= last_5pct)]   # נחלק את הדאטה לבדיקה, מה שמעל האינדקס של ה5% העליונים הוא לבקרה
main_df = new_df[(new_df.index < last_5pct)]               # ושאר ה95% של הנתונים הוא לאימון המודל

train_x, train_y = preprocess_df(main_df)     # עיבוד הדאטה מטבלה לסדרות ולייבלים לאימון המודל
validation_x, validation_y = preprocess_df(validation_main_df)  # עיבוד הדאטה מטבלה לסדרות ולייבלים לווידוא המודל

# to see the train and validation data:
print(train_x)
print(train_x[0])

print(f"train data: {len(train_x)} validation: {len(validation_x)}")                       #  ווידוא כמות נתונים בסט הנתונים לאימון
print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")                         # בדיקת כמה קונים וכמה מוכרים
print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")    # בדיקת כמות מוכרים וקונים בסט הוואלידציה

print(train_x[0].shape)
print(train_x.shape[1:])
print(type(train_x))
#preprocess_df(main_df)


#The model:

"""

model = Sequential()  # יצירת ארכיטקטורה למודל
model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))  # שכבת LSTM ראשונה
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape=(train_x.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation="relu"))    # שכבה מלאה דאנס - עם הונקציה 'רילו'
model.add(Dropout(0.2))

train_y = tf.keras.utils.to_categorical(train_y)      # להפוך את הלייבלים למטריצה של ווקטורי 1-הוט =זא מטריצת יחידה שמתאימה לייבל i לעמודה הi
validation_y = tf.keras.utils.to_categorical(validation_y)   # להפוך את הלייבלים למטריצה של ווקטורי 1-הוט =זא מטריצת יחידה שמתאימה לייבל i לעמודה הi

model.add(Dense(2, activation="softmax"))   # שכבה אחרונה ומלאה - דאנס עם פונקציית ,הסופטמאקס' כי אנו בפלט וצריך לאזן


optimizer = Adam(learning_rate=0.001)     # כאן מתחיל להגדיר את המודל לפי keras כאשר ככל שהערך למידה קטן יותר זה גורר למידה טובה יותר
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])   # כאן מגדירים את האופטימיציטור, פונקציית הלוס והמטריקה שאיתה נעבוד במודל

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

filepath = "RNN_Final-{epoch:02d}-{val_accuracy:.3f}.model"
checkpoint = ModelCheckpoint("models/{}.model".format(filepath),
                              monitor='val_accuracy',
                              verbose=1,
                              save_best_only=True,
                              mode='max')
# Train model
train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(BATCH_SIZE)
validation_dataset = tf.data.Dataset.from_tensor_slices((validation_x, validation_y)).batch(BATCH_SIZE)

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    callbacks=[tensorboard, checkpoint])


# Score model
score = model.evaluate(validation_dataset, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Save model
model.save("models/{}".format(NAME))

"""


def cut_data(df,start=62,to=1):
    test = df[(df.index >= (len(times)-int(start)))]  # יצירת טבלת נתונים של ה61 ימים האחרונים
    test = test.drop(test.index[-int(to)])   # מחיקת השורה האחרונה
    return test

test = cut_data(new_df, 62, 1)
test_x = creating_test(test)   # עיבוד הנתונים שנרצה לנתח שבה המודל יודע לעבוד

print(test_x)
print(test_x[0].shape)

# Load saved model
model = load_model("models/{}".format(NAME))    #  טעינת המודל שבנינו ואימנו לפני

# Make predictions on new data
predictions = model.predict(test_x)    #  הפעלת המודל על המבחן שאנו רוצים לחזות ושמירת התוצאות

# Print predictions
print(predictions)
# The `predictions` variable will contain the predicted probabilities for each class for each data point in `test_x`. You can use `np.argmax()` to get the predicted class for each data point.
#print(predicted_classes = np.argmax(predictions, axis=1))


