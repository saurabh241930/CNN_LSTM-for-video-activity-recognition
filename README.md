# CNN-LSTM-for-Activity-Recognition

**This method is use to detect particular activity in certain span of video**

This works by passing spatial-temporal features of video frames which is extracted  by state of the art Inception network and passing into **LSTM** network

![](https://i.imgur.com/6zQdCkV.jpg)

## Setup

## Breaking down code 

Pre-processing code

Importing extractor model 

Here we are using inceptionv3 model removing its last layer and using n-1th layer as output

```python
base_model = InceptionV3(weights='imagenet',include_top=True)
extactor_model = Model(inputs=base_model.input,outputs=base_model.get_layer('avg_pool').output)
```

Conversion code

```python
def pre_process(video_dir):
    classes = []
    X = []
    y = []
    all_videos_path = video_dir


    if not os.path.exists('images'):
        os.makedirs('images')

    for fld in os.listdir(all_videos_path):
        classes.append(fld)

    for fld in os.listdir(all_videos_path):
        for file in os.listdir(all_videos_path+fld):
            video_path = str(all_videos_path+fld+'/'+file)
            video_length = float(str(getLength(video_path)).split(",")[0].split(":")[-1])

            if video_length < 1:
                print("[WARN] Skipping video because of wrong length")
                pass
            else:
                imagesFolder = "images"
                cap = cv2.VideoCapture(video_path)
                frameRate = cap.get(5)
                count = 0

                if video_length < 1.5:
                    controller = 4
                elif video_length < 2.5:
                    controller = 6
                elif video_length < 3.5:
                    controller = 8
                elif video_length < 5.5:
                    controller = 10
                elif video_length < 8.5:
                    controller = 12
                else:
                    controller = 12

                folder = '/tf/CNN_LSTM/images'
                for the_file in os.listdir(folder):
                    file_path = os.path.join(folder, the_file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        print(e)


                while(cap.isOpened()):
                    frameId = cap.get(1) 
                    ret, frame = cap.read()
                    if (ret != True):
                        break
                    if (frameId%controller==0):
                        count = count + 1
                        filename = imagesFolder + "/image_" +  str(int(count)) + ".jpg"
                        cv2.imwrite(filename, frame)

                    if frameId == controller*6:
                        break
                cap.release()
                print ("[INFO] Video Converted")

                files = []
                for f in os.listdir('images'):
                    if f.endswith(".jpg"):
                        files.append("images/"+str(f))

                sequence = []

                for f in files:
                    img = image.load_img(f, target_size=(299, 299))
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x)
                    features = extactor_model.predict(x)
                    sequence.append(features)

                label_encoded = classes.index(str(fld))
                label_hot = to_categorical(label_encoded, len(classes))


                X.append(sequence)
                y.append(label_hot)
                print("[INFO] Features Extracted for {} Frames at controller {} for class {}".format(len(sequence),controller,fld))

    np.save("X_data",X)            
    np.save("y_data",y)
```




```
Output should look something like this

```
BabyCrawling: 0.99
Archery: 0.00
ApplyEyeMakeup: 0.00
ApplyLipstick: 0.00
```


