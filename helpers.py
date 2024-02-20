import time
import openai
from sklearn.metrics import confusion_matrix, precision_score, f1_score, roc_auc_score
import pandas as pd

def getPrediction(text):
    return 1 if 'Outcome=1' in text else (0 if 'Outcome=0' in text else None)

def find_majority_element(nums):
    element_count = {}
    for num in nums:
        element_count[num] = element_count.get(num, 0) + 1

    majority_element = max(element_count, key=element_count.get) if element_count else None

    return majority_element if majority_element is not None and element_count[majority_element] > len(nums) // 2 else "tie"

def get_gpt_response(messages, model="gpt-3.5-turbo-16k-0613"): 
    try:
        response = openai.ChatCompletion.create(model=model, messages=messages)
        return response.choices[0].message["content"]
    except Exception as e:
        print("Error", e)

def gpt_prompt(messages):
    reply = None
    counter = 0
    while counter < 5 and reply is None:
        reply = get_gpt_response(messages)
        counter += 1
        if counter == 5:
            time.sleep(1)
    return reply

def calculate_metrics(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    
    if len(np.unique(y_true))>1:
        auc = roc_auc_score(y_true, y_pred)
        metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'F1 Score': f1,
        'Specificity': specificity,
        'Sensitivity': sensitivity,
        'AUC': auc
        }
    else:
        # Create a dictionary of the calculated metrics
        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'F1 Score': f1,
            'Specificity': specificity,
            'Sensitivity': sensitivity            
        }

    return metrics

def extract_text_after_rules(text):
    # Find the index of "rules="
    index = text.find("Rules=")
    if index == -1:
        return ("")
        #return None  # "rules=" not found in the text

    # Extract the substring after "rules="
    substring = text[index + len("Rules="):]

    return substring


association_rules_perf = pd.DataFrame(
    {
        "run": [],
        "nrules": [],
        "fitnes": [],
        "support": [],
        "confidence": []
    }
)

logreg_perf = pd.DataFrame(
    {
        "run": [],
        "Accuracy": [],
        "Precision": [],
        "F1 Score": [],
        "Specificity": [],
        "Sensitivity": [],
        "AUC": []
    }
)

gpt_perf = pd.DataFrame(
    {
        "run": [],
        "Accuracy": [],
        "Precision": [],
        "F1 Score": [],
        "Specificity": [],
        "Sensitivity": [],
        "AUC": []
    }
)

gpt_explanations = pd.DataFrame(
    {
        "run": [],
        "output": [],
        "rules": []
    }
)