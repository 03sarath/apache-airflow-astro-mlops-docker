from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import numpy as np

# Define paths to store the model and data
MODEL_PATH = '/usr/local/airflow/models/model.pkl'  # Change path as needed
DATA_PATH = '/usr/local/airflow/data/data.csv'      # Change path as needed

# Ensure model and data directories exist
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

# Functions for tasks
def data_prep(**kwargs):
    print("Performing Data Preparation")
    # Load your dataset
    # For example, let's say you're using the Iris dataset for illustration
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the data into training/testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Prepare the prediction data (could be another dataset or just the test set)
    X_predict = X_test  # This could be a different dataset or a specific subset

    # Push the prepared data to XCom
    kwargs['ti'].xcom_push(key='train_test_data', value={
        'X_train': X_train.tolist(),
        'y_train': y_train.tolist(),
        'X_test': X_test.tolist(),
        'y_test': y_test.tolist(),
        'X_predict': X_predict.tolist()  # Add this line to include prediction data
    })


def model_train(**kwargs):
    print("Performing Model training process")
    # Fetch XCom directly within the function
    raw_data = kwargs['ti'].xcom_pull(task_ids='data_prep', key='train_test_data')

    # Convert lists back to NumPy arrays
    X_train = np.array(raw_data['X_train'])
    y_train = np.array(raw_data['y_train'])

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, MODEL_PATH)
    kwargs['ti'].xcom_push(key='model_path', value=MODEL_PATH)

def model_eval(**kwargs):
    # Fetch the model and evaluation data directly using XCom
    model_path = kwargs['ti'].xcom_pull(task_ids='model_train', key='model_path')
    raw_data = kwargs['ti'].xcom_pull(task_ids='data_prep', key='train_test_data')

    # Convert lists back to NumPy arrays
    X_test = np.array(raw_data['X_test'])
    y_test = np.array(raw_data['y_test'])

    # Load the model
    model = joblib.load(model_path)

    # Evaluate the model
    score = model.score(X_test, y_test)
    
    print(f"Model evaluation score: {score}")
    kwargs['ti'].xcom_push(key='model_score', value=score)


def model_prediction(**kwargs):
    # Fetch the model and the data for prediction
    model_path = kwargs['ti'].xcom_pull(task_ids='model_train', key='model_path')
    raw_data = kwargs['ti'].xcom_pull(task_ids='data_prep', key='train_test_data')

    # Ensure the key exists in the raw_data
    if 'X_predict' not in raw_data:
        raise KeyError("Key 'X_predict' not found in raw_data.")

    # Convert the data back to NumPy arrays for prediction
    X_predict = np.array(raw_data['X_predict'])

    # Load the trained model
    model = joblib.load(model_path)

    # Make predictions
    print("Performing Model predictions")
    predictions = model.predict(X_predict)
    
    # Push predictions to XCom
    kwargs['ti'].xcom_push(key='model_predictions', value=predictions.tolist())  # Convert to list for XCom



def model_deploy(**kwargs):
    # Pull predictions from XCom
    predictions = kwargs['ti'].xcom_pull(task_ids='model_prediction', key='model_predictions')
    
    # Here you would implement your deployment logic.
    # For demonstration, we'll just print the predictions
    print("Model Predictions:")
    for prediction in predictions:
        print(prediction)

    # Implement additional logic for deploying the model (e.g., saving to a server)


# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),  # Set to your desired start date
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG('ml_pipeline_op_kwargs',
         default_args=default_args,
         schedule_interval='*/30 * * * *',  # Run every 30 minutes
         catchup=False) as dag:

    # Task to prepare data and push to XCom
    task_data_prep = PythonOperator(
        task_id='data_prep',
        python_callable=data_prep,
        provide_context=True
    )

    # Task to train the model, pulling data from XCom
    task_model_train = PythonOperator(
    task_id='model_train',
    python_callable=model_train,
    provide_context=True
)


    # Task to evaluate the model, pulling model path and data from XCom
    task_model_eval = PythonOperator(
    task_id='model_eval',
    python_callable=model_eval,
    provide_context=True
)


    # Task to make predictions
    task_model_prediction = PythonOperator(
    task_id='model_prediction',
    python_callable=model_prediction,
    provide_context=True,
    dag=dag
)

    task_model_deploy = PythonOperator(
    task_id='model_deploy',
    python_callable=model_deploy,
    provide_context=True,
    dag=dag
)

    # Define task dependencies
    task_data_prep >> task_model_train >> task_model_eval >> task_model_prediction >> task_model_deploy
