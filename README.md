# Multimedia-Retrieval
This project is made in python 3.8.5, certain libraries are not available in later versions. Therefor it is recommended to run it with this verion. All the required libraries are present in the requirements.txt. With the command below all libraries will be installed:

```pip install -r requirements.txt```

To run the project:

```python app.py```

Take care that this file creates two processes, one is the streamlit ui. This ui depends on the Flask api process. It might take some time before both are running. It shouldn't take more then a couple seconds for the Flask api to start up in the background. 

Extra code that was used to evaluate the program, and other can be found inside the `extra-code` folder.