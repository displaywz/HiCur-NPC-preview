import pandas as pd
import gradio as gr

# 创建一个用户的全局变量
current_user = None

# 定义所有医生/模型选项
all_doctors = ['A', 'B', 'C', 'D', 'E', 'F']

# 定义加载用户CSV文件的函数
def load_user_data(user_id):
    global df, ids
    
    # 构造用户的CSV文件路径
    file_path = f'./hicur/{user_id}_patients_data.csv'
    
    # 读取用户的CSV文件，如果不存在则返回False
    try:
        df = pd.read_csv(file_path)
        ids = df['id'].tolist()  # 获取所有患者的id
        return True  # 文件存在并且加载成功
    except FileNotFoundError:
        return False  # 文件不存在

# 获取患者数据并更新界面
def update_patient_data(selected_id, doctor):
    row = df[df['id'] == selected_id].iloc[0]
    clinical_record = row['clinical_record']
    img_file = row['imgfile']
    
    # 根据选中的医生/模型更新诊疗方案和评分
    treatment_plan = row[f'treatment_plan_{doctor}']
    accuracy = row[f'accuracy_{doctor}'] if not pd.isna(row[f'accuracy_{doctor}']) else 0
    completeness = row[f'completeness_{doctor}'] if not pd.isna(row[f'completeness_{doctor}']) else 0
    reliability = row[f'reliability_{doctor}'] if not pd.isna(row[f'reliability_{doctor}']) else 0
    safety = row[f'safety_{doctor}'] if not pd.isna(row[f'safety_{doctor}']) else 0
    
    return (clinical_record, treatment_plan, accuracy, completeness, reliability, safety, 
            img_file, selected_id, gr.update(value=doctor))

# 保存评分并更新CSV文件
def save_scores(selected_id, accuracy, completeness, reliability, safety, doctor):
    df.loc[df['id'] == selected_id, [f'accuracy_{doctor}', f'completeness_{doctor}', f'reliability_{doctor}', f'safety_{doctor}']] = [
        accuracy, completeness, reliability, safety
    ]
    
    file_path = f'./hicur/{current_user}_patients_data.csv'
    df.to_csv(file_path, index=False)
    
    return "打分保存成功！"

# 获取前一个未评分的患者ID和医生
def get_previous_unscored(current_id):
    current_index = ids.index(current_id)
    for i in range(current_index - 1, -1, -1):
        patient_id = ids[i]
        unscored_doctor = get_next_unscored_doctor(patient_id)
        if unscored_doctor is not None:
            return patient_id, unscored_doctor
    # 如果没有找到，从最后一个开始搜索
    for i in range(len(ids) - 1, current_index, -1):
        patient_id = ids[i]
        unscored_doctor = get_next_unscored_doctor(patient_id)
        if unscored_doctor is not None:
            return patient_id, unscored_doctor
    return ids[0], all_doctors[0]  # 如果所有都已评分，返回第一个患者和医生A

# 获取下一个未评分的患者ID和医生
def get_next_unscored(current_id):
    current_index = ids.index(current_id)
    for i in range(current_index, len(ids)):
        patient_id = ids[i]
        unscored_doctor = get_next_unscored_doctor(patient_id)
        if unscored_doctor is not None:
            return patient_id, unscored_doctor
    # 如果没有找到，从开头开始搜索
    for i in range(0, current_index):
        patient_id = ids[i]
        unscored_doctor = get_next_unscored_doctor(patient_id)
        if unscored_doctor is not None:
            return patient_id, unscored_doctor
    return ids[0], all_doctors[0]  # 如果所有都已评分，返回第一个患者和医生A

def login_user(user_id):
    global current_user
    if load_user_data(user_id):  # 如果用户文件存在
        current_user = user_id
        
        # 查找第一个包含未评分的记录，从索引0开始
        first_unscored_id, first_unscored_doctor = get_next_unscored(ids[0])
        
        return (gr.update(value=user_id),  # 清除错误信息
                gr.update(choices=ids),  # 更新ID列表
                *update_patient_data(first_unscored_id, first_unscored_doctor))  # 加载未评分的患者数据
    else:
        # 用户不存在时，返回错误提示
        return (gr.update(value="用户不存在", visible=True), 
                gr.update(choices=[]),  # 清空ID列表
                "", "", 0, 0, 0, 0, "", None, gr.update(value=all_doctors[0]))  # 清空其他字段

def get_next_unscored_doctor(patient_id):
    for doctor in all_doctors:
        if df.loc[df['id'] == patient_id, [f'accuracy_{doctor}', f'completeness_{doctor}', f'reliability_{doctor}', f'safety_{doctor}']].sum().sum() == 0:
            return doctor
    return None  # 如果所有医生都已评分，返回None

# Gradio界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            # 用户登录框
            user_id_input = gr.Textbox(label="Enter user ID to log in")
            login_button = gr.Button("Log in")

            # ID选择栏，显示所有ID
            id_list = gr.Dropdown(choices=[], label="Select patient ID")

            previous_button = gr.Button("Previous")
            next_button = gr.Button("Next")

            # 更新医生/模型单选框
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
    
    # 用户登录后，加载相应的数据和ID列表
    login_button.click(
        login_user,
        inputs=[user_id_input],
        outputs=[user_id_input, id_list, clinical_record, treatment_plan, 
                 accuracy, completeness, reliability, safety,
                 patient_image, id_list, doctor_selection]
    )
    
    # ID选择后的数据更新
    id_list.change(
        lambda selected_id: update_patient_data(selected_id, get_next_unscored_doctor(selected_id) or all_doctors[0]),
        inputs=[id_list], 
        outputs=[clinical_record, treatment_plan, 
                 accuracy, completeness, reliability, safety,
                 patient_image, id_list, doctor_selection]
    )
    
    # 当选择不同医生/模型时更新诊疗方案和评分
    doctor_selection.change(
        lambda doctor, selected_id: update_patient_data(selected_id, doctor),
        inputs=[doctor_selection, id_list],
        outputs=[clinical_record, treatment_plan, 
                 accuracy, completeness, reliability, safety,
                 patient_image, id_list, doctor_selection]
    )
    
    # 提交打分动作
    submit_button.click(
        lambda selected_id, accuracy, completeness, reliability, safety, doctor: save_scores(selected_id, accuracy, completeness, reliability, safety, doctor), 
        inputs=[id_list, accuracy, completeness, reliability, safety, doctor_selection], 
        outputs=score_output
    )
    
    # 上一个按钮的操作
    previous_button.click(
        lambda x: update_patient_data(*get_previous_unscored(x)),
        inputs=[id_list],
        outputs=[clinical_record, treatment_plan, 
                 accuracy, completeness, reliability, safety,
                 patient_image, id_list, doctor_selection]
    )
    
    # 下一个按钮的操作
    next_button.click(
        lambda x: update_patient_data(*get_next_unscored(x)),
        inputs=[id_list],
        outputs=[clinical_record, treatment_plan, 
                 accuracy, completeness, reliability, safety,
                 patient_image, id_list, doctor_selection]
    )

# 启动 Gradio
demo.launch()