import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d

dfold = pd.read_pickle("./preprocessing/dataframe.pkl")
df = pd.read_pickle("./preprocessing/dataframe_resampled_pml.pkl")
dfpml2 = pd.read_pickle("./dataframe_resampledPML2.pkl")
st.dataframe(df, width=600, height=300, use_container_width=True)  # Custom width and height
st.dataframe(dfpml2, width=600, height=300, use_container_width=True)  # Custom width and height

user_input = st.text_input("Enter object path", "./resampledPML/Jet/m1147.obj")

if st.button("Show mesh before transformation"):
    # st.write("Showing: ", user_input)
    mesh = o3d.io.read_triangle_mesh("."+user_input.replace("resampledPML", "database"))
    window = o3d.visualization.draw_geometries([mesh], width=1280, height=720, mesh_show_wireframe=True)
    
if st.button("Show mesh after transformation PML"):
    # st.write("Showing: ", user_input)
    mesh = o3d.io.read_triangle_mesh("."+user_input)
    window = o3d.visualization.draw_geometries([mesh], width=1280, height=720, mesh_show_wireframe=True)

if st.button("Show mesh after transformation O3D"):
    # st.write("Showing: ", user_input)
    mesh = o3d.io.read_triangle_mesh("."+user_input.replace("resampledPML", "resampledO3D"))
    window = o3d.visualization.draw_geometries([mesh], width=1280, height=720, mesh_show_wireframe=True)

if st.button("Show mesh after transformation PML2"):
    # st.write("Showing: ", user_input)
    mesh = o3d.io.read_triangle_mesh("."+user_input.replace("resampledPML", "resampledPML2"))
    window = o3d.visualization.draw_geometries([mesh], width=1280, height=720, mesh_show_wireframe=True)

if st.button("Show histograms before transformation"):
    # Create a histogram
    fig, (ax1, ax2)  = plt.subplots(1, 2, figsize=(12, 4))  # Create subplots with 1 row and 2 columns

    ax1.hist(dfold['Vertices'], bins=100, edgecolor='k')  # Adjust the number of bins as needed
    ax1.set_xlabel('Vertices')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Histogram of Vertices')
    ax1.grid(True)
    # Display the histogram
    # plt.show()
    # Create a histogram
    ax2.hist(dfold['Triangles'], bins=100, edgecolor='k')  # Adjust the number of bins as needed
    ax2.set_xlabel('Triangles')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Histogram of Triangles')
    ax2.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

if st.button("Show histograms after transformation"):

    fig, (ax1, ax2)  = plt.subplots(1, 2, figsize=(12, 4))  # Create subplots with 1 row and 2 columns
    ax1.hist(df['Vertices'], bins=100, edgecolor='k')  # Adjust the number of bins as needed
    ax1.set_xlabel('Vertices')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Histogram of Vertices')
    ax1.grid(True)
    # Display the histogram
    # plt.show()
    # Create a histogram
    ax2.hist(df['Triangles'], bins=100, edgecolor='k')  # Adjust the number of bins as needed
    ax2.set_xlabel('Triangles')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Histogram of Triangles')
    ax2.grid(True)
    plt.tight_layout()
    st.pyplot(fig)