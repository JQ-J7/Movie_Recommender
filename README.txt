========================================================================================
         TARUMT - ARTIFICIAL INTELLIGENCE (AI) GROUP PROJECT
              Hybrid Movie Recommender System - README
========================================================================================

PROJECT OVERVIEW
----------------
This is a Streamlit web application that recommends movies using a Hybrid approach:
  - Content-Based Filtering (CBF): TF-IDF vectorization + Cosine Similarity on
    genres, keywords, cast, director, and synopsis.
  - Collaborative Filtering (CF): Pearson Correlation on user-item rating patterns.
  - Hybrid Engine: Weighted blend of CBF and CF scores (alpha hyperparameter).

The app has two views:
  - User View     : Search movies and get Top-N recommendations.
  - Developer View: Password-protected (PIN: 1234). Tune hyperparameters, run
                    offline 80/20 evaluation metrics, and audit survey responses.

----------------------------------------------------------------------------------------
PROJECT FILES
----------------------------------------------------------------------------------------
  app.py                        - Main Streamlit web application (entry point)
  module_hybrid.py              - Hybrid recommender engine & evaluation logic
  collaborative_recommender.py  - Collaborative Filtering (CF) engine
  content_based_recommender.py  - Content-Based Filtering (CBF) engine
  movies_dataset.csv            - MovieLens dataset (required, ~79 MB)
  survey_responses.csv          - User satisfaction survey responses

========================================================================================
STEP 1 - PREREQUISITES: INSTALL PYTHON
========================================================================================

Make sure Python 3.9 or higher is installed on your machine.
Download from: https://www.python.org/downloads/

To verify, open a terminal / command prompt and run:
    python --version

========================================================================================
STEP 2 - INSTALL REQUIRED PACKAGES
========================================================================================

Open a terminal in the project folder and run the following command to install
all required libraries at once:

    pip install streamlit pandas numpy scikit-learn

Individual packages explained:
  - streamlit    : Web application framework (runs the app in the browser)
  - pandas       : Data loading and manipulation
  - numpy        : Numerical computations
  - scikit-learn : TF-IDF, Cosine Similarity, train_test_split, MSE/RMSE metrics

To verify installations:
    pip show streamlit pandas numpy scikit-learn

========================================================================================
STEP 3 - HOW TO RUN THE APP
========================================================================================

------------------------------------------------------------------------
OPTION A: RUNNING IN VS CODE
------------------------------------------------------------------------

1. Open VS Code.
2. Open the project folder:
       File > Open Folder > select the "Movie_Recommender" folder

3. Open the VS Code integrated terminal:
       View > Terminal   (or press Ctrl + `)

4. Make sure you are inside the project folder in the terminal.
   You should see the path ending in "Movie_Recommender".

5. Run the Streamlit app:
       streamlit run app.py

6. The app will automatically open in your default browser at:
       http://localhost:8501

   If it does not open automatically, copy and paste the URL into your browser.

7. To stop the app, go back to the terminal and press:
       Ctrl + C

TIP: Install the "Python" and "Pylance" extensions in VS Code for better
     Python support and syntax highlighting.

------------------------------------------------------------------------
OPTION B: RUNNING ON GOOGLE COLAB
------------------------------------------------------------------------

NOTE: Google Colab is a cloud notebook environment. Running Streamlit on Colab
requires a tunnel (like localtunnel or pyngrok) because Colab does not expose
ports directly. Follow these steps:

1. Upload your project files to Google Drive:
   - movies_dataset.csv
   - survey_responses.csv
   - app.py
   - module_hybrid.py
   - collaborative_recommender.py
   - content_based_recommender.py

2. Open a new Google Colab notebook: https://colab.research.google.com/

3. Mount Google Drive and go to the project folder:
       from google.colab import drive
       drive.mount('/content/drive')
       %cd /content/drive/MyDrive/Movie_Recommender

4. Install required packages:
       !pip install streamlit pandas numpy scikit-learn pyngrok -q

5. Set up a free ngrok account at https://ngrok.com/ and get your auth token.
   Then authenticate:
       from pyngrok import ngrok
       ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN_HERE")

6. Run Streamlit in the background and create a public tunnel:
       import threading, subprocess
       def run():
           subprocess.run(["streamlit", "run", "app.py",
                           "--server.port=8501", "--server.headless=true"])
       t = threading.Thread(target=run, daemon=True)
       t.start()

       public_url = ngrok.connect(8501)
       print("App is live at:", public_url)

7. Click the printed URL to open the app in your browser.

NOTE: The large movies_dataset.csv (~79 MB) may take a while to upload to Drive.
      Make sure all 6 project files are in the same folder on Drive.

------------------------------------------------------------------------
OPTION C: RUNNING STREAMLIT DIRECTLY (COMMAND PROMPT / POWERSHELL)
------------------------------------------------------------------------

1. Open Command Prompt or PowerShell.

2. Navigate to the project folder:
       cd path\to\Movie_Recommender

   Example:
       cd D:\Movie_Recommender

3. Run the app:
       streamlit run app.py

4. Open your browser and go to:
       http://localhost:8501

5. To stop the app, press Ctrl + C in the terminal.

========================================================================================
TROUBLESHOOTING
========================================================================================

PROBLEM : 'streamlit' is not recognized as a command
SOLUTION: Python's Scripts folder is not in your system PATH.
          Try running with:
              python -m streamlit run app.py
          Or add Python's Scripts folder to your PATH environment variable.

PROBLEM : ModuleNotFoundError: No module named 'streamlit' (or other packages)
SOLUTION: Install the missing package:
              pip install <package-name>
          If you have multiple Python versions, use:
              pip3 install streamlit pandas numpy scikit-learn

PROBLEM : FileNotFoundError: movies_dataset.csv not found
SOLUTION: Make sure movies_dataset.csv is in the SAME folder as app.py.
          The app will not run without the dataset file.

PROBLEM : Port 8501 is already in use
SOLUTION: Run on a different port:
              streamlit run app.py --server.port 8502

PROBLEM : App is very slow to load on first run
SOLUTION: This is expected. The app builds TF-IDF matrices and similarity structures
          from the dataset on first load (~79 MB). Subsequent loads use cached data
          and will be much faster. Do not close the terminal while loading.

========================================================================================
DEVELOPER VIEW ACCESS
========================================================================================

To access the Developer & Evaluation Studio:
  1. Click "Developer View" (or equivalent toggle) in the app.
  2. Enter PIN: 1234

Features available in Developer View:
  - Hybrid Engine alpha weighting hyperparameter tuning
  - 80/20 offline evaluation metrics (MSE, RMSE, Precision@10, Recall@10, F1@10)
  - Content-Based Filtering standalone evaluation
  - Collaborative Filtering standalone evaluation
  - Survey response auditing and analytics
  - System cache control (Clear Cache & Reload Data)

========================================================================================
QUICK REFERENCE - INSTALL & RUN (COPY-PASTE READY)
========================================================================================

    # Install all dependencies
    pip install streamlit pandas numpy scikit-learn

    # Run the app (from inside the Movie_Recommender folder)
    streamlit run app.py

========================================================================================
