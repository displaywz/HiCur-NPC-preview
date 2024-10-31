import pandas as pd
import gradio as gr

# Create a global variable for the user
current_user = None

# Define all doctor/model options
all_doctors = ['A', 'B', 'C', 'D', 'E', 'F']

# Define a function to load the user's CSV file
def load_user_data(user_id):
    global df, ids
    
    # Construct the file path for the user's CSV
    file_path = f'./hicur/{user_id}_patients_data.csv'
    
    # Read the user's CSV file; if it doesn't exist, return False
    try:
        df = pd.read_csv(file_path)
        ids = df['id'].tolist()  # Get all patient IDs
        return True  # File exists and loads successfully
    except FileNotFoundError:
        return False  # File does not exist

# Retrieve patient data and update the interface
def update_patient_data(selected_id, doctor):
    row = df[df['id'] == selected_id].iloc[0]
    clinical_record = row['clinical_record']
    img_file = row['imgfile']
    
    # Update treatment plan and scores based on the selected doctor/model
    treatment_plan = row[f'treatment_plan_{doctor}']
    accuracy = row[f'accuracy_{doctor}'] if not pd.isna(row[f'accuracy_{doctor}']) else 0
    completeness = row[f'completeness_{doctor}'] if not pd.isna(row[f'completeness_{doctor}']) else 0
    reliability = row[f'reliability_{doctor}'] if not pd.isna(row[f'reliability_{doctor}']) else 0
    safety = row[f'safety_{doctor}'] if not pd.isna(row[f'safety_{doctor}']) else 0
    
    return (clinical_record, treatment_plan, accuracy, completeness, reliability, safety, 
            img_file, selected_id, gr.update(value=doctor))

# Save scores and update the CSV file
def save_scores(selected_id, accuracy, completeness, reliability, safety, doctor):
    df.loc[df['id'] == selected_id, [f'accuracy_{doctor}', f'completeness_{doctor}', f'reliability_{doctor}', f'safety_{doctor}']] = [
        accuracy, completeness, reliability, safety
    ]
    
    file_path = f'./hicur/{current_user}_patients_data.csv'
    df.to_csv(file_path, index=False)
    
    return "Score saved successfully!"

# Get the previous unscored patient ID and doctor
def get_previous_unscored(current_id):
    current_index = ids.index(current_id)
    for i in range(current_index - 1, -1, -1):
        patient_id = ids[i]
        unscored_doctor = get_next_unscored_doctor(patient_id)
        if unscored_doctor is not None:
            return patient_id, unscored_doctor
    # If none is found, start searching from the last entry
    for i in range(len(ids) - 1, current_index, -1):
        patient_id = ids[i]
        unscored_doctor = get_next_unscored_doctor(patient_id)
        if unscored_doctor is not None:
            return patient_id, unscored_doctor
    return ids[0], all_doctors[0]  # If all are scored, return the first patient and Doctor A

# Get the next unscored patient ID and doctor
def get_next_unscored(current_id):
    current_index = ids.index(current_id)
    for i in range(current_index, len(ids)):
        patient_id = ids[i]
        unscored_doctor = get_next_unscored_doctor(patient_id)
        if unscored_doctor is not None:
            return patient_id, unscored_doctor
    # If none is found, start searching from the beginning
    for i in range(0, current_index):
        patient_id = ids[i]
        unscored_doctor = get_next_unscored_doctor(patient_id)
        if unscored_doctor is not None:
            return patient_id, unscored_doctor
    return ids[0], all_doctors[0]  # If all are scored, return the first patient and Doctor A

def login_user(user_id):
    global current_user
    if load_user_data(user_id):  # If the user file exists
        current_user = user_id
        
        # Find the first unscored record from index 0
        first_unscored_id, first_unscored_doctor = get_next_unscored(ids[0])
        
        return (gr.update(value=user_id),  # Clear error message
                gr.update(choices=ids),  # Update ID list
                *update_patient_data(first_unscored_id, first_unscored_doctor))  # Load unscored patient data
    else:
        # Return an error message if the user does not exist
        return (gr.update(value="User does not exist", visible=True), 
                gr.update(choices=[]),  # Clear ID list
                "", "", 0, 0, 0, 0, "", None, gr.update(value=all_doctors[0]))  # Clear other fields

def get_next_unscored_doctor(patient_id):
    for doctor in all_doctors:
        if df.loc[df['id'] == patient_id, [f'accuracy_{doctor}', f'completeness_{doctor}', f'reliability_{doctor}', f'safety_{doctor}']].sum().sum() == 0:
            return doctor
    return None  # Return None if all doctors are scored

# Gradio Interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            # User login box
            user_id_input = gr.Textbox(label="Enter user ID to log in")
            login_button = gr.Button("Log in")

            # ID selection field, displaying all IDs
            id_list = gr.Dropdown(choices=[], label="Select patient ID")

            previous_button = gr.Button("Previous")
            next_button = gr.Button("Next")

            # Update doctor/model radio button
            doctor_selection = gr.Radio(choices=all_doctors, label="Select doctor/model", value=all_doctors[0])

        with gr.Column(scale=3):
            clinical_record = gr.Textbox(label="Standard", lines=5)
            treatment_plan = gr.Textbox(label="Model output", lines=5)
            
            accuracy = gr.Slider(0, 10, step=1, label="Accuracy")
            completeness = gr.Slider(0, 10, step=1, label="Completeness")
            reliability = gr.Slider(0, 10, step=1, label="Reliability")
            safety = gr.Slider(0, 10, step=1, label="Safety")
            
            patient_image = gr.Image(label="Patient image")
            
            submit_button = gr.Button("Submit Score")
            score_output = gr.Textbox(label="Scoring Result", lines=2)
    
    # Load relevant data and ID list after user login
    login_button.click(
        login_user,
        inputs=[user_id_input],
        outputs=[user_id_input, id_list, clinical_record, treatment_plan, 
                 accuracy, completeness, reliability, safety,
                 patient_image, id_list, doctor_selection]
    )
    
    # Update data after ID selection
    id_list.change(
        lambda selected_id: update_patient_data(selected_id, get_next_unscored_doctor(selected_id) or all_doctors[0]),
        inputs=[id_list], 
        outputs=[clinical_record, treatment_plan, 
                 accuracy, completeness, reliability, safety,
                 patient_image, id_list, doctor_selection]
    )
    
    # Update treatment plan and scores when selecting a different doctor/model
    doctor_selection.change(
        lambda doctor, selected_id: update_patient_data(selected_id, doctor),
        inputs=[doctor_selection, id_list],
        outputs=[clinical_record, treatment_plan, 
                 accuracy, completeness, reliability, safety,
                 patient_image, id_list, doctor_selection]
    )
    
    # Submit scoring action
    submit_button.click(
        lambda selected_id, accuracy, completeness, reliability, safety, doctor: save_scores(selected_id, accuracy, completeness, reliability, safety, doctor), 
        inputs=[id_list, accuracy, completeness, reliability, safety, doctor_selection], 
        outputs=score_output
    )
    
    # Action for the previous button
    previous_button.click(
        lambda x: update_patient_data(*get_previous_unscored(x)),
        inputs=[id_list],
        outputs=[clinical_record, treatment_plan, 
                 accuracy, completeness, reliability, safety,
                 patient_image, id_list, doctor_selection]
    )
    
    # Action for the next button
    next_button.click(
        lambda x: update_patient_data(*get_next_unscored(x)),
        inputs=[id_list],
        outputs=[clinical_record, treatment_plan, 
                 accuracy, completeness, reliability, safety,
                 patient_image, id_list, doctor_selection]
    )

# Launch Gradio
demo.launch()
