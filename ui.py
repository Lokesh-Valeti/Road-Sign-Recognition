import streamlit as st
from ultralytics import YOLO
from PIL import Image
import traceback
from pathlib import Path
 


st.title("Road sign detection")
model = YOLO(str(Path.cwd())+"\\runs\\detect\\train6\\weights\\best.pt")
names= {0: 'Green Light, ', 1: 'Red Light, ', 2: 'Speed Limit 10,', 3: 'Speed Limit 100,', 
        4: 'Speed Limit 110,', 5: 'Speed Limit 120,', 6: 'Speed Limit 20,', 7: 'Speed Limit 30,', 
        8: 'Speed Limit 40,', 9: 'Speed Limit 50,', 10: 'Speed Limit 60,', 11: 'Speed Limit 70,', 12: 'Speed Limit 80,',
          13: 'Speed Limit 90,', 14: 'Stop, '}
stoutput=["You can proceed safely","You have to stop at traffic signal","Set your speed limit to 10",
          "Set your speed limit to 100","Set your speed limit to 110","Set your speed limit to 120",
          "Set your speed limit to 20","Set your speed limit to 30","Set your speed limit to 40",
          "Set your speed limit to 50","Set your speed limit to 60","Set your speed limit to 70",
          "Set your speed limit to 80","Set your speed limit to 90","Stop your Vehicle"]
with st.form("user input"):
    uploaded_file=st.file_uploader('Upload an image', type=['jpg', 'png', 'jpeg'])

    button=st.form_submit_button("Get the output")

    if button and uploaded_file:
        with st.spinner("loading..."):
            try:
                # Preprocess the image
                image = Image.open(uploaded_file)
                predictions=model.predict(source=image)[0].verbose()
                output=" ".join(predictions.split(" ")[1:4])
                for key, value in names.items():
                    if value == output:
                        idx=key
                st.text_area(label="Output ", value=stoutput[idx])

            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error(e)