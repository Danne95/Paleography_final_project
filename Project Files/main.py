#   http://35.195.54.187:8501/

#pipreqs ./ --force #for requirements      #needed for cloud
# change opencv to this!!!: opencv-python-headless<=[Num]

#run site after closing terminal ==> nohup command 
#to find all processes running ==> ps -aux
#kill process ==> kill -9 'PID'

# bash commands for streamlit-server
#terminal:
#streamlit run main.py
#streamlit run paleo_site/main.py
#nohup streamlit run yourscript.py
#streamlit run paleo_site/main.py --server.port=80


# imports
import streamlit as st
from PIL import Image
import myConfig.predict as myPred
import cv2
import numpy as np
from hashlib import sha256
import os
from math import ceil, floor
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from datetime import datetime
import pytz
import csv


# consts
users_folder = 'myConfig/users/'
accounts_file = 'myConfig/accounts.txt'
fieldnames = ['proxy_name', 'filename', 'time', 'SCE_origin', 'SCE_origin_accuracy', 'SCE_style', 'SCE_style_accuracy', 
'BGU_origin', 'BGU_origin_accuracy', 'BGU_style', 'BGU_style_accuracy', 'user_origin', 'user_style']
#datetime.now(pytz.timezone('Asia/Jerusalem')) #get time is israel

# session state -> keeps values on script rerun.
if 'script_run_count' not in st.session_state:
    st.session_state['script_run_count'] = 0
if 'page_login_status' not in st.session_state:
    st.session_state['page_login_status'] = False
if 'user_name' not in st.session_state:
    st.session_state['user_name'] = 'None'
if 'saved_flag' not in st.session_state:
    st.session_state['saved_flag'] = False
if 'user_ans' not in st.session_state:
    st.session_state['user_ans'] = 'None'   

def sign_up_func(username, password):
    if '~~~' in username or  '~~~' in password:
        st.sidebar.error("'~'  is a restricted symbol and cannot be used in the input")#alert
        return
    acc_file = open(accounts_file,'r+')
    for line in acc_file.readlines():
        if line.split('~~~')[0] == username:
            st.sidebar.error("Username is already taken")#alert
            acc_file.close()
            return
    acc_file.write(username + '~~~' + (sha256(password.encode('utf-8')).hexdigest()) +'\n')
    try:
        os.mkdir(path=users_folder+username)
    except FileExistsError:
        pass
    with open(users_folder + username + '/' + 'history.csv', 'w') as csvfile:
        global fieldnames
        writer = csv.DictWriter(csvfile)
        csvfile.close()
    st.sidebar.success("Account created successfully")#alert
    acc_file.close()

def login_func(username, password):
    if '~~~' in username or  '~~~' in password:
        st.sidebar.error("'~'  is a restricted symbol and cannot be used in the input")#alert
        return
    acc_file = open(accounts_file,'r+')
    for line in acc_file.readlines():
        if line.split('~~~')[0] == username:
            if line.split('~~~')[1] == (sha256(password.encode('utf-8')).hexdigest() +'\n'):
                st.session_state.page_login_status = True
                st.session_state['user_name'] = username
                st.sidebar.success("Logged in successfully")
            else:
                st.sidebar.error("Login failed (wrong password)")
            acc_file.close()
            return
    st.sidebar.error("Log in failed (user not found)")
    acc_file.close()

def logout_func():
    st.session_state.page_login_status = False
    st.session_state['user_name'] = 'None'

def on_model_select():
    if st.session_state['choice'] != ' ':
        with rows[2][0]:
            st.subheader("Step 3:click the predict button to the left")
        with rows[2][1]:
            st.button(label="Predict", on_click=start_prediction, args = (img,image_file.name))

def start_prediction(image_file, image_name):
    image_file_2 = cv2.cvtColor(np.array(image_file), cv2.COLOR_RGB2BGR)
    st.header("Prediction:")
    if st.session_state['choice'] == 'Both':
        prediction1 = myPred.predict_image(image_file_2,myPred.class_model,myPred.subclass_model)
        prediction2 = myPred.predict_image(image_file_2,myPred.class_model,myPred.subclass_model)
        with rows[3][0]:
            # SCE
            SCE_pred_name = myPred.revert((prediction1["class"],prediction1["subclass"]))
            SCE_pred_class_acc = max(prediction1["class_bins"])*100
            SCE_pred_subclass_acc = max(prediction1["subclass_bins"])*100
            st.write("SCE model think \'{}\' is: \n".format(image_name)+SCE_pred_name)
            st.write("Class confidence: {:.2f}\n".format(SCE_pred_class_acc))
            st.write("Sublass confidence: {:.2f}\n".format(SCE_pred_subclass_acc))
            # BGU_
            BGU_pred_name = myPred.revert((prediction2["class"],prediction2["subclass"]))
            BGU_pred_class_acc = max(prediction2["class_bins"])*100
            BGU_pred_subclass_acc = max(prediction2["subclass_bins"])*100
            st.write("BGU model  think \'{}\' is: \n".format(image_name)+BGU_pred_name)
            st.write("Class confidence: {:.2f}\n".format(BGU_pred_class_acc))
            st.write("Sublass confidence: {:.2f}\n".format(BGU_pred_subclass_acc))

        # option to save classification
        if st.session_state['page_login_status'] == True:
            with rows[3][1]:
                with st.form("input_class_form"):
                    st.write("Choose your classification of the image:")
                    st.selectbox(label='Classification' ,key="user_ans", options=("None", "ashkenazi cursive", "italian cursive", "sephardic cursive",
                    "ashkenazi semisquare", "byzantine semisquare", "italian semisquare", "oriental semisquare", 
                    "sephardic semisquare", "yemenite semisquare", "ashkenazi square", "byzantine square", 
                    "italian square", "oriental square", "sephardic square", "yemenite square"))
                    st.form_submit_button("save this prediction", on_click=save_process, args=(image_file,image_name,
                    [SCE_pred_name,SCE_pred_class_acc,SCE_pred_subclass_acc],
                    [BGU_pred_name,BGU_pred_class_acc,BGU_pred_subclass_acc]))
    
    # one model
    else: 
        # SCE
        if st.session_state['choice'] == 'SCE':
            prediction = myPred.predict_image(image_file_2,myPred.class_model,myPred.subclass_model)
        # BGU
        else:
            #display SCE results while waiting for BGU files.
            prediction = myPred.predict_image(image_file_2,myPred.class_model,myPred.subclass_model)
        with rows[3][0]:
            pred_name = myPred.revert((prediction["class"],prediction["subclass"]))
            pred_class_acc = max(prediction["class_bins"])*100
            pred_subclass_acc = max(prediction["subclass_bins"])*100
            st.write("We think \'{}\' is: \n".format(image_name)+pred_name)
            st.write("Origin confidence: {:.2f}\n".format(pred_class_acc))
            st.write("Style confidence: {:.2f}\n".format(pred_subclass_acc))

# save uploaded image as smaller thumbnail, and save prediction data to user's file
def save_process(image_file, image_name, SCE_pred, BGU_pred):
    # save image thumbnail to user's folder 
    resized_image_to_save = image_file.copy()
    resized_image_to_save.thumbnail(size=(400,400))
    isr_time = (datetime.now(pytz.timezone('Asia/Jerusalem'))).strftime("%m/%d/%Y, %H:%M:%S")
    isr_time_os = (datetime.now(pytz.timezone('Asia/Jerusalem'))).strftime("#%d%m%Y--%H:%M:%S#")
    proxy_name = users_folder + st.session_state['user_name'] + '/' + isr_time_os + image_name
    resized_image_to_save.save(proxy_name)

    # save data to csv
    user_ans = st.session_state['user_ans']
    if user_ans == 'None':
        user_ans = 'None None'
    history_line = [proxy_name, image_name, isr_time, SCE_pred[0].split(' ')[0], SCE_pred[1], SCE_pred[0].split(' ')[1], SCE_pred[2],
    BGU_pred[0].split(' ')[0], BGU_pred[1], BGU_pred[0].split(' ')[1], BGU_pred[2], user_ans.split(' ')[0], user_ans.split(' ')[1]]
    with open(users_folder + st.session_state['user_name'] + '/' + 'history.csv', 'a') as csvfile:
        writer_object = csv.writer(csvfile)
        writer_object.writerow(history_line)
        csvfile.close()

    # end statements 
    st.session_state['saved_flag'] = True
    st.success("Successfully saved to your history..")

def display_history():
    with open(users_folder + st.session_state['user_name'] + '/' + 'history.csv', 'r') as csvfile:
        reader_object = csv.reader(csvfile)
        counter = 0
        row_count = len(list(reader_object))
        csvfile.seek(0)
        disp_grid = [st.columns([1,1,1]) ]* ceil(row_count/3)
        for row in reader_object:
            with disp_grid[floor(counter/3)][counter%3]:
                st.image(row[0], use_column_width = True)
                st.write("Image name: ", row[1])
                st.write("Upload time: ", row[2])
                st.write("SCE model preidiction was {} {}".format(row[3], row[5]))
                st.write("Certainty: ({:.2f},{:.2f})".format(float(row[4]),float(row[6])))
                st.write("BGU model preidiction was {} {}".format(row[7], row[9]))
                st.write("Certainty: ({:.2f},{:.2f})".format(float(row[8]),float(row[10])))
                if row[11] == 'None':
                    tmp = 'Not decided'
                else:
                    tmp = str(row[11] + ' ' + row[12])
                st.write("Your classification was ", tmp)
            counter += 1
    return

# page initialization
st.set_page_config(page_title="Palaeography Classification", page_icon=":crystal_ball:", layout="wide") #has to be first line!
st.session_state['script_run_count'] +=1 # helper (does nothing)
st.title("Hebrew paleography classifier web application!")
st.write("[Learn more about paleography>](https://en.wikipedia.org/wiki/Palaeography)")
menu = ["Login", "Sign Up", "History"]
side_choice = st.sidebar.selectbox("user menu:", menu, key='sidebar_choice')
rows = [st.columns([3,4]),st.columns([3,4]),st.columns([3,4]),st.columns([3,4])]
print(st.session_state)

if side_choice == "History":
    #logged out
    if st.session_state['page_login_status'] == False:
        st.subheader("welcome to the user history page!")
        st.write("Currently you are a 'guest' user")
        st.write("History is collected only for registered users while logged in")
        st.write("History is being collected only with user agreement")
        st.write("If you are interested in the history feature, please consider registering")
    #logged in    
    else:
        st.subheader("Welcome to the user history page, {}!".format(st.session_state['user_name']))
        display_history()
        st.write("End of history")


# choice is "Login" or "Sign Up"
else:
    with rows[0][0]:
        st.header("Instructions:")
        st.subheader("Step 1:upload a document image")

    with rows[0][1]:
        st.header("Process:")
        image_file = st.file_uploader("Upload Image:", type=["png","jpg","jpeg"])
        if image_file is not None:
            with rows[0][0]:
                st.image("/home/historicalmanuscripts/paleo_site/scrolldown.gif") #arrow pointing down gif
            # process image file
            img = Image.open(image_file)
                
            # To view uploaded image
            st.image(img, use_column_width = True)
            with rows[1][1]:
                global choice
                choice = st.radio("Model selection:", (' ', 'SCE', 'BGU', 'Both'),on_change =on_model_select, key='choice')
                if st.session_state['saved_flag'] == True:
                    st.success("Successfully saved to your history")
                    st.session_state['saved_flag'] = False 
            with rows[1][0]:
                st.subheader("Step 2:choose a model from the radio buttons")

    if side_choice == "Login":
        #true (logged-in)
        if st.session_state['page_login_status']:
            st.sidebar.write("Welcome ", st.session_state['user_name'])
            st.sidebar.button(label='Log out', on_click=logout_func)
        
        #false (not logged)
        else:
            username = st.sidebar.text_input(label="User Name")
            password = st.sidebar.text_input(label="Password", type='password')
            st.sidebar.button(label='log in', on_click=login_func, args=(username,password))

    elif side_choice == "Sign Up":
        username = st.sidebar.text_input(label="User Name")
        password = st.sidebar.text_input(label="Password", type='password',)
        st.sidebar.button(label="Sign Up", on_click=sign_up_func, args=(username,password))

