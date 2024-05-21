# Medicine_Predictor_with_Streamlit

# Medicine Dosage and Prescription Predictor

This project is designed to predict medicine dosage and prescriptions based on input data using machine learning models.

## Table of Contents
- [Forking the Repository](#forking-the-repository)
- [Installing dependencies](#Installing-Dependencies)
- [Setting Up the Environment](#setting-up-the-environment)
- [Running the Project](#running-the-project)
- [Project Structure](#project-structure)
- [License](#license)

## Forking the Repository

To fork the repository, follow these steps:([forking steps](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo))

1. Navigate to the repository page on GitHub.
2. Click the "Fork" button at the top-right corner of the page.
3. Choose your GitHub account to fork the repository to.

Once you have forked the repository, you can clone it to your local machine:

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```
## Installing Dependencies:
```bash
pip install -r requirements.txt
```

## Setting Up the Environment:(if required)

```bash
python -m venv venv
source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
```
## Running the Project:
```bash
python run ml_model.py # To generate new .pkl files for the ML model.
streamlit run app.py
```
## Project Structure
```bash
.
├── app.py                     # Main application file
├── Dataset.csv                # Dataset file used for training the models
├── dosage_predictor.pkl       # Pre-trained dosage predictor model
├── label_encoders.pkl         # Label encoders for categorical variables
├── medicine_predictor.pkl     # Pre-trained medicine predictor model
├── ml_model.py                # Script to train and evaluate models
├── requirements.txt           # List of dependencies
└── README.md                  # Project documentation
```

## License
```bash

Feel free to adjust any sections to better fit your project's specifics or additional information you would like to include.

```
