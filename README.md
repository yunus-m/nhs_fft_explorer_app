# nhs_fft_explorer_app
A desktop application for processing and exploring NHS FFT sentiment data

## Usage overview

## Running the app
In each case below, point your browser to `http://localhost:8501` to view the app.

### Directly from a Python environment
Install the `requirements.txt` file using `pip` or set up an equivalent environment.

Launch the app using `streamlit run streamlit_app.py`

### Build and run using Docker
Build the Docker file using `docker build . -t fft_explorer_app`.

Launch the app using `docker run -p 8501:8501 fft_explorer_app:latest`

### PyInstaller for deploying a platform-independent app
The user will need an installation of Docker Desktop.
Save the Docker image out using `docker save -o fft_explorer_app.tar fft_explorer_app:latest`

Compile the app on your target platform (Windows, Mac, or Linux); for Linux I use:
`pyinstaller --onefile app_launcher.py --add_data "data:data" --add-data "fft_explorer_image.tar:."

The compiled version will internally invoke `app_launcher.py` which loads the Docker image and starts a container.