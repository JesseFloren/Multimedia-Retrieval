import pyvista as pv
import streamlit as st
from stpyvista import stpyvista
import requests
import os
import glob
import extra_streamlit_components as stx
from datetime import datetime
import shutil

pv.global_theme.show_scalar_bar = False
st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 370px !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if "path" not in st.session_state:
    st.session_state['path'] = None
    st.session_state['type'] = None
    st.session_state['query'] = False
    st.session_state['method'] = "feature"


with st.sidebar:
    st.header("Query Settings")
    chosen_id = stx.tab_bar(data=[
        stx.TabBarItemData(id="upload", title="Upload", description=""),
        stx.TabBarItemData(id="database", title="Database", description=""),
    ], default="upload")

    if chosen_id == "upload":
        if st.session_state['type'] != "File":
            st.session_state['path'] = None
            st.session_state['type'] = "File"
        
        uploaded_file = st.file_uploader("Choose a file")

        if uploaded_file is not None:
            with open("./target.obj", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state['path'] = "./target.obj"

            st.subheader("Query Object")
            plotter = pv.Plotter(window_size=[200,200])
            mesh = pv.read(st.session_state['path'])
            plotter.add_mesh(mesh, cmap='bwr', line_width=1)
            plotter.view_isometric()
            plotter.background_color = 'black'
            stpyvista(plotter, key="target", horizontal_align="left")

    if chosen_id == "database":
        if st.session_state['type'] != "Feature":
            st.session_state['path'] = None
            st.session_state['type'] = "Feature"
        
        dbpath = "./featuresPML2"
        folders = os.listdir(dbpath)

        class_name = st.selectbox('Class', folders)
        if class_name:
            class_folder_path = os.path.join(dbpath, class_name)
            files = glob.glob(os.path.join(class_folder_path, '*'))
            file_name = st.selectbox('Object', ([file.replace(class_folder_path + "\\", "") for file in files]))
            b21 = st.button('Choose Object')
            if b21: 
                st.session_state['path'] = class_folder_path + "\\" + file_name
                st.subheader("Query Object")

                plotter = pv.Plotter(window_size=[200,200])
                mesh = pv.read((class_folder_path + "\\" + file_name + ".obj").replace("featuresPML2", "database"))
                plotter.add_mesh(mesh, cmap='bwr', line_width=1)
                plotter.view_isometric()
                plotter.background_color = 'black'
                stpyvista(plotter, key="target2", horizontal_align="left")

st.title("3D Objects Retrieval System")

st.header("Results")
path = st.session_state['path']
type = st.session_state['type']

def callback():
    st.session_state['query'] = True


if path is not None and not st.session_state['query']:
    st.session_state['method'] = st.sidebar.selectbox('Query Method', ("custom", "tsne"))
    q = st.sidebar.button('Run Query', on_click=callback)


cols = st.columns(10, gap="small")

st.header("Previous")

@st.cache_data
def display_images(dirs):
    if len(dirs) > 5:
        for dir in dirs[:-5]:
            shutil.rmtree("./previous/" + dir)
            dirs.remove(dir)
    dirs.reverse()
    for folder in dirs:
        cols2 = st.columns(10, gap="small")
        files = glob.glob(os.path.join("./previous/" + folder, '*svg'))
        for i in range(10):
            with cols2[i]:
                st.image(os.path.abspath(files[i]))

dirs = os.listdir("./previous/")
display_images(dirs)


if path is not None and st.session_state['query']:
    st.session_state['query'] = False
    res = None
    if type == "Feature":
        res = requests.post("http://127.0.0.1:5000/" + st.session_state['method'], json={"path": path}).json()
    else:
        res = requests.post("http://127.0.0.1:5000", json={"path": path}).json()


    currentDateAndTime = datetime.now()
    currentTime = currentDateAndTime.strftime("%Y-%m-%d_%H-%M-%S")
    prev_path = "./previous/" + currentTime
    os.mkdir(prev_path)

    col = 0
    for _, p in res["results"]:
        plotter = pv.Plotter(window_size=[180,180])
        obj_path = p.replace("featuresPML2", "database") + ".obj"
        mesh = pv.read(obj_path)
        plotter.add_mesh(mesh, cmap='bwr', line_width=1)
        plotter.view_isometric()
        plotter.background_color = 'gray'
        plotter.save_graphic(prev_path + "/{}".format(col) + ".svg")
        plotter.background_color = 'black'
        with cols[col]:
            stpyvista(plotter, key=p, horizontal_align="left")
        col += 1