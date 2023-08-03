import streamlit as st
import numpy as np
from PIL import Image
import time
from scipy.spatial.distance import cdist

@st.cache_data
def read_data():
    all_vecs = np.load("all_vecs.npy" , allow_pickle=True)
    all_names = np.load("all_names.npy", allow_pickle=True)
    return all_vecs , all_names

vecs , names = read_data()

_ , fcol2 , _ = st.columns(3)

scol1 , scol2 = st.columns(2)

ch = scol1.button("Start / change")
fs = scol2.button("find similar")

      
if ch:
    if len(names) > 0:
        random_name = names[np.random.randint(len(names))]
        fcol2.image(Image.open("./images_original/" + random_name))
        st.session_state["disp_img"] = random_name
        st.write(st.session_state["disp_img"])
    else:
        st.warning("No images available.")
if fs:
    c1 , c2 , c3 , c4 , c5 = st.columns(5)
    
    if st.session_state.disp_img is not None:
        idx = np.where(names == st.session_state.disp_img)[0][0]
        target_vec = vecs[idx]
    fcol2.image(Image.open("./images_original/" + st.session_state["disp_img"]))
    top5 = cdist(target_vec[None , ...] , vecs).squeeze().argsort()[1:6]
    c1.image(Image.open("./images_original/" + names[top5[0]]))
    c2.image(Image.open("./images_original/" + names[top5[1]]))
    c3.image(Image.open("./images_original/" + names[top5[2]]))
    c4.image(Image.open("./images_original/" + names[top5[3]]))
    c5.image(Image.open("./images_original/" + names[top5[4]]))










    
