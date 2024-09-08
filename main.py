import argparse
import pickle
import re
import ast
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import joblib
from catboost import CatBoostRegressor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BERTClass(nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.roberta = AutoModel.from_pretrained("models/rubert-tiny2")
        self.fc = nn.Linear(312, 248)

    def forward(self, ids, mask, token_type_ids):
        _, features = self.roberta(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        output = self.fc(features)
        return output


class BERTDataset(Dataset):
    def __init__(self, X, tokenizer, max_len):
        self.len = len(X)
        self.X = X.reset_index(drop=True)
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        text = self.X.iloc[index]
        inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        }

def classificatoin(X_test):
    # Impute nans with empty string for simplicity
    imputer = SimpleImputer(fill_value='', strategy='constant')
    X_test = pd.DataFrame(imputer.fit_transform(X_test), columns = X_test.columns)

    # Concatenate the features
    X_test_concat = X_test.demands + ' ' + X_test.company_name + ' ' +  X_test.achievements_modified

    # Load the encoder
    oe = pickle.load(open('models/classification/ordinal_encoder.pkl', 'rb'))
    encoder = pickle.load(open('models/classification/one_hot_encoder.pkl', 'rb'))

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("models/rubert-tiny2")

    # Load the model
    model = BERTClass()
    model.load_state_dict(torch.load('models/classification/job_name_model.pt'))
    model.to(device)

    # Create the dataloader
    valid_dataset = BERTDataset(X_test_concat, tokenizer, 256)
    valid_loader = DataLoader(valid_dataset, batch_size=64, num_workers=4, shuffle=False, pin_memory=True)

    # Compute the model predictions
    model.eval()
    y_preds = []
    with torch.no_grad():
        for _, data in enumerate(valid_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids)
            _, preds = torch.max(outputs, dim = 1)
            y_preds.extend(preds)
    y_preds_processed = np.array([])

    for tens in y_preds:
        tens = tens.cpu().numpy()
        y_preds_processed = np.append(y_preds_processed, tens)

    # Create the dataframe with the predictions
    classification_result_df = pd.DataFrame(columns = ['id', 'job_name', 'task_type'])
    classification_result_df['id'] = X_test['id']
    classification_result_df['job_name'] = oe.inverse_transform(y_preds_processed.reshape(-1, 1)).ravel()
    classification_result_df['task_type'] = 'RES'

    return classification_result_df


encoders = {
    'academic_degree': joblib.load('models/regression/academic_degree_encoder.joblib'),
    'accommodation_type': joblib.load('models/regression/accommodation_type_encoder.joblib'),
    'bonus_type': joblib.load('models/regression/bonus_type_encoder.joblib'),
    'measure_type': joblib.load('models/regression/measure_type_encoder.joblib'),
    'busy_type': joblib.load('models/regression/busy_type_encoder.joblib'),
    'education': joblib.load('models/regression/education_encoder.joblib'),
    'original_source_type': joblib.load('models/regression/original_source_type_encoder.joblib'),
    'company_business_size': joblib.load('models/regression/company_business_size_encoder.joblib'),
    'schedule_type': joblib.load('models/regression/schedule_type_encoder.joblib'),
    'social_protected_ids': joblib.load('models/regression/social_protected_ids_encoder.joblib'),
    'source_type': joblib.load('models/regression/source_type_encoder.joblib'),
    'state_region_code': joblib.load('models/regression/state_region_code_encoder.joblib'),
    'status': joblib.load('models/regression/status_encoder.joblib'),
    'transport_compensation': joblib.load('models/regression/transport_compensation_encoder.joblib'),
    'vacancy_benefit_ids': joblib.load('models/regression/vacancy_benefit_ids_encoder.joblib'),
    'professionalSphereName': joblib.load('models/regression/professionalSphereName_encoder.joblib'),
    'federalDistrictCode': joblib.load('models/regression/federalDistrictCode_encoder.joblib'),
    'code_professional_sphere': joblib.load('models/regression/code_professional_sphere_encoder.joblib'),
    'required_drive_license': joblib.load('models/regression/required_drive_license_encoder.joblib'),
    'languageKnowledge': joblib.load('models/regression/languageKnowledge_encoder.joblib')
}

def tokenize(vacancy_name):
    # Tokenize the vacancy name using regular expressions
    return set(re.split(r'[^\w]+', vacancy_name.lower()))

def compute_token_intersections(tokens1, tokens2):
    # Compute the intersection score between two sets of tokens
    return len(tokens1 & tokens2)

tokens_dict_profession = {}
code_profession_dict = {}


def impute_values(df, is_train, train_df=None):
    global tokens_dict_profession, code_profession_dict
    if is_train:
        # Handling 'code_profession'
        non_nan_train = train_df[train_df['code_profession'].notna()]
        vacancy_to_code_profession = non_nan_train.set_index('vacancy_name')['code_profession'].to_dict()
        df['code_profession'] = df['vacancy_name'].map(vacancy_to_code_profession).fillna(df['code_profession'])

        for index, row in df[df['code_profession'].notna()].iterrows():
            tokens_dict_profession[index] = tokenize(row['vacancy_name'])
            code_profession_dict[index] = row['code_profession']

        for index, row in df[df['code_profession'].isna()].iterrows():
            current_tokens = tokenize(row['vacancy_name'])
            max_intersection = 0
            best_match = None

            for other_index, other_tokens in tokens_dict_profession.items():
                intersection_score = compute_token_intersections(current_tokens, other_tokens)

                if intersection_score > max_intersection:
                    max_intersection = intersection_score
                    best_match = code_profession_dict[other_index]
                if intersection_score == 2 or intersection_score == len(current_tokens):
                    break

            df.at[index, 'code_profession'] = best_match

        df['code_profession'].fillna(df['code_profession'].median(), inplace=True)

        # Train set imputation for 'code_professional_sphere'
        non_nan_train = train_df[train_df['code_professional_sphere'].notna()]
        vacancy_to_code_professional_sphere = non_nan_train.set_index('vacancy_name')['code_professional_sphere'].to_dict()

        df['code_professional_sphere'] = df['vacancy_name'].map(vacancy_to_code_professional_sphere).fillna(df['code_professional_sphere'])

        tokens_dict_professional_sphere = {}
        code_professional_sphere_dict = {}

        for index, row in df[df['code_professional_sphere'].notna()].iterrows():
            tokens_dict_professional_sphere[index] = tokenize(row['vacancy_name'])
            code_professional_sphere_dict[index] = row['code_professional_sphere']

        for index, row in df[df['code_professional_sphere'].isna()].iterrows():
            current_tokens = tokenize(row['vacancy_name'])
            max_intersection = 0
            best_match = None

            for other_index, other_tokens in tokens_dict_professional_sphere.items():
                intersection_score = compute_token_intersections(current_tokens, other_tokens)

                if intersection_score > max_intersection:
                    max_intersection = intersection_score
                    best_match = code_professional_sphere_dict[other_index]
                if intersection_score == 2 or intersection_score == len(current_tokens):
                    break

            df.at[index, 'code_professional_sphere'] = best_match

        return df

    else:
        # Test set imputation based on train_df
        if train_df is None:
            raise ValueError("train_df must be provided for test set imputation")

        # Handling 'code_profession'
        non_nan_train = train_df[train_df['code_profession'].notna()]
        vacancy_to_code_profession = non_nan_train.set_index('vacancy_name')['code_profession'].to_dict()

        df['code_profession'] = df['vacancy_name'].map(vacancy_to_code_profession)
        #print(df['code_profession'])
        #df['code_profession'].fillna(df['code_profession'].apply(
        #    lambda x: find_best_match(x, tokens_dict_profession, code_profession_dict)), inplace=True)
        # Handling 'code_professional_sphere'
        non_nan_train = train_df[train_df['code_professional_sphere'].notna()]
        vacancy_to_code_professional_sphere = non_nan_train.set_index('vacancy_name')['code_professional_sphere'].to_dict()

        df['code_professional_sphere'] = df['vacancy_name'].map(vacancy_to_code_professional_sphere)
        #df['code_professional_sphere'].fillna(df['code_professional_sphere'].apply(
        #    lambda x: find_best_match(x, tokens_dict_professional_sphere, code_professional_sphere_dict)), inplace=True)

        return df

def find_best_match(vacancy_name, tokens_dict, code_dict):
    current_tokens = tokenize(vacancy_name)
    max_intersection = 0
    best_match = None

    for other_index, other_tokens in tokens_dict.items():
        intersection_score = compute_token_intersections(current_tokens, other_tokens)

        if intersection_score > max_intersection:
            max_intersection = intersection_score
            best_match = code_dict[other_index]
        if intersection_score == 2 or intersection_score == len(current_tokens):
            break
    return best_match

categories_which_need_medcard = set()

def needs_medcard(category):
    return category in categories_which_need_medcard

def hash_col(df, col, N):
    cols = [col + "_" + str(i) for i in range(N)]

    # Preallocate a zero matrix
    result = np.zeros((df.shape[0], N), dtype=int)

    # Apply hash function and set the corresponding column to 1
    for idx, val in enumerate(df[col]):
        result[idx, hash(val) % N] = 1

    # Create DataFrame from the result matrix
    df_hash = pd.DataFrame(result, columns=cols)

    # Concatenate the new columns to the original DataFrame and drop the old column
    return pd.concat([df.drop(col, axis=1), df_hash], axis=1)


def encode_column_train(df, column_name, prefix):
    encoders[column_name].fit(df[[column_name]])
    encoded = encoders[column_name].transform(df[[column_name]])

    encoded_df = pd.DataFrame(encoded, columns=encoders[column_name].get_feature_names([prefix]))

    # Ensure the encoded DataFrame has the same index as the original
    encoded_df.index = df.index

    return encoded_df


def encode_column(df, column_name, prefix):
    encoded = encoders[column_name].transform(df[[column_name]])

    categories = encoders[column_name].categories_[0]  # Assuming one feature
    feature_names = [f"{prefix}_{cat}" for cat in categories]
    encoded_df = pd.DataFrame(encoded, columns=feature_names)
    #encoded_df = pd.DataFrame(encoded, columns=encoders[column_name].get_feature_names([prefix]))
    # Ensure the encoded DataFrame has the same index as the original
    encoded_df.index = df.index

    return encoded_df


def explode_and_encode_train(df, column_name, prefix, value=None):
    df[column_name] = df[column_name].apply(ast.literal_eval)
    if value is not None:
        exploded = df[column_name].apply(lambda x: [d[value] for d in x]).explode()
    else:
        exploded = df[column_name].explode()
    encoded = encoders[column_name].fit_transform(exploded.values.reshape(-1, 1))
    encoded_df = pd.DataFrame(encoded, columns=encoders[column_name].get_feature_names([prefix]))
    return encoded_df.groupby(level=0).sum()


def explode_and_encode(df, column_name, prefix, value=None):
    df[column_name] = df[column_name].apply(ast.literal_eval)
    if value is not None:
        exploded = df[column_name].apply(lambda x: [d[value] for d in x]).explode()
    else:
        exploded = df[column_name].explode()
    encoded = encoders[column_name].transform(exploded.values.reshape(-1, 1))
    encoded_df = pd.DataFrame(encoded, columns=encoders[column_name].get_feature_names([prefix]))
    return encoded_df.groupby(level=0).sum()


def fillna_with_mode(df):
    for col in df.columns:
        if df[col].dtype in ['number', 'object', 'category']:
            # Calculate the mode, excluding NaN
            mode_value = df[col].mode()[0] if not df[col].mode().empty else None

            # Fill NaN values with the mode value
            df[col].fillna(mode_value, inplace=True)

    return df

premium_scaler = joblib.load('models/regression/premium_scaler.joblib')
experience_scaler = joblib.load('models/regression/experience_scaler.joblib')
salary_scaler = joblib.load('models/regression/salary_scaler.joblib')
places_scaler = joblib.load('models/regression/places_scaler.joblib')


def preprocess(df, is_train: bool = True, train_df: pd.DataFrame = None):
    global categories_which_need_medcard, premium_scaler, experience_scaler, salary_scaler, places_scaler
    # Drop unnecessary columns
    df = df.drop('id', axis=1)
    df = df.drop('contact_person', axis=1)
    df = df.drop('deleted', axis=1)
    df = df.drop('metro_ids', axis=1)
    df = df.drop('is_moderated', axis=1)
    df = df.drop('regionNameTerm', axis=1)
    df = df.drop('company_name', axis=1)
    df = df.drop('contact_source', axis=1)
    df = df.drop('data_ids', axis=1)
    df = df.drop('foreign_workers_capability', axis=1)
    df = df.drop('is_uzbekistan_recruitment', axis=1)
    df = df.drop('oknpo_code', axis=1)
    df = df.drop('okso_code', axis=1)
    df = df.drop('publication_period', axis=1)
    df = df.drop('contactList', axis=1)
    df = df.drop('regionName', axis=1)
    df = df.drop('retraining_capability', axis=1)
    df = df.drop('retraining_condition', axis=1)
    df = df.drop('retraining_grant_value', axis=1)
    df = df.drop('vacancy_address_additional_info', axis=1)
    df = df.drop('vacancy_address', axis=1)
    df = df.drop('vacancy_address_code', axis=1)
    df = df.drop('vacancy_address_house', axis=1)
    df = df.drop('vacancy_address_latitude', axis=1)
    df = df.drop('vacancy_address_longitude', axis=1)
    df = df.drop('industryBranchName', axis=1)
    df = df.drop('full_company_name', axis=1)
    df = df.drop('company_inn', axis=1)
    df = df.drop('company', axis=1)
    df = df.drop('company_code', axis=1)
    df = df.drop('code_external_system', axis=1)
    df = df.drop('visibility', axis=1)
    df = df.drop('additional_requirements', axis=1)
    df = df.drop('education_speciality', axis=1)
    df = df.drop('other_vacancy_benefit', axis=1)
    df = df.drop('position_requirements', axis=1)
    df = df.drop('position_responsibilities', axis=1)
    df = df.drop('required_certificates', axis=1)
    df = df.drop('hardSkills', axis=1)
    df = df.drop('softSkills', axis=1)
    df = df.drop('required_drive_license', axis=1)
    df = df.drop('languageKnowledge', axis=1)

    ## Trying without date
    df = df.drop('change_time', axis=1)
    df = df.drop('date_create', axis=1)
    df = df.drop('date_modify', axis=1)
    df = df.drop('published_date', axis=1)

    # Deal with NaN
    df['academic_degree'].fillna('absent', inplace=True)
    df['accommodation_type'].fillna('absent', inplace=True)
    # df['additional_requirements'].fillna('absent', inplace=True)
    df['bonus_type'].fillna('absent', inplace=True)
    df['measure_type'].fillna('absent', inplace=True)
    # df['education_speciality'].fillna('absent', inplace=True)
    # df['other_vacancy_benefit'].fillna('absent', inplace=True)
    # df['position_requirements'].fillna('absent', inplace=True)
    # df['position_responsibilities'].fillna('absent', inplace=True)
    # df['required_certificates'].fillna('absent', inplace=True)
    df['social_protected_ids'].fillna('absent', inplace=True)
    df['transport_compensation'].fillna('absent', inplace=True)
    df['vacancy_benefit_ids'].fillna('absent', inplace=True)
    df['professionalSphereName'].fillna('absent', inplace=True)
    df['is_mobility_program'].fillna(False, inplace=True)
    df['additional_premium'].fillna(0, inplace=True)
    df['required_experience'].fillna(0, inplace=True)
    if is_train:
        df = impute_values(df, is_train=True, train_df=df)
    else:
        df = impute_values(df, is_train=False, train_df=train_df)
    categories_which_need_medcard = set(df[df['need_medcard'] == True]['code_professional_sphere'])
    df['need_medcard'] = df['need_medcard'].fillna(df['code_professional_sphere'].apply(needs_medcard))

    ### Transform
    df['accommodation_capability'] = df['accommodation_capability'].astype(int)
    df['career_perspective'] = df['career_perspective'].astype(int)
    df['is_mobility_program'] = df['is_mobility_program'].astype(int)
    df['is_quoted'] = df['is_quoted'].astype(int)
    df['need_medcard'] = df['need_medcard'].astype(int)
    df['retraining_grant'] = df['retraining_grant'].replace({'нет стипендии': 0, 'есть стипендия': 1})
    n_features = 100
    if is_train:
        df = df.dropna(subset=['state_region_code'])
        df = df.dropna(subset=['federalDistrictCode'])
        df = df.loc[df['salary'] <= 400000]
        df = pd.concat(
            [df.drop('academic_degree', axis=1), encode_column_train(df, 'academic_degree', 'academic_degree')], axis=1)
        df = pd.concat([df.drop('accommodation_type', axis=1),
                        encode_column_train(df, 'accommodation_type', 'accommodation_type')], axis=1)
        df = pd.concat([df.drop('bonus_type', axis=1), encode_column_train(df, 'bonus_type', 'bonus_type')], axis=1)
        df = pd.concat([df.drop('measure_type', axis=1), encode_column_train(df, 'measure_type', 'measure_type')],
                       axis=1)
        df = pd.concat([df.drop('busy_type', axis=1), encode_column_train(df, 'busy_type', 'busy_type')], axis=1)
        df = pd.concat([df.drop('education', axis=1), encode_column_train(df, 'education', 'education')], axis=1)
        df = pd.concat([df.drop('original_source_type', axis=1),
                        encode_column_train(df, 'original_source_type', 'original_source_type')], axis=1)
        df = pd.concat([df.drop('company_business_size', axis=1),
                        encode_column_train(df, 'company_business_size', 'company_business_size')], axis=1)
        df = pd.concat([df.drop('schedule_type', axis=1), encode_column_train(df, 'schedule_type', 'schedule_type')],
                       axis=1)
        df = pd.concat([df.drop('social_protected_ids', axis=1),
                        encode_column_train(df, 'social_protected_ids', 'social_protected_ids')], axis=1)
        df = pd.concat([df.drop('source_type', axis=1), encode_column_train(df, 'source_type', 'source_type')], axis=1)
        df = pd.concat(
            [df.drop('state_region_code', axis=1), encode_column_train(df, 'state_region_code', 'state_region_code')],
            axis=1)
        df = pd.concat([df.drop('status', axis=1), encode_column_train(df, 'status', 'status')], axis=1)
        df = pd.concat([df.drop('transport_compensation', axis=1),
                        encode_column_train(df, 'transport_compensation', 'transport_compensation')], axis=1)
        df = pd.concat([df.drop('vacancy_benefit_ids', axis=1),
                        encode_column_train(df, 'vacancy_benefit_ids', 'vacancy_benefit_ids')], axis=1)
        df = pd.concat([df.drop('professionalSphereName', axis=1),
                        encode_column_train(df, 'professionalSphereName', 'professionalSphereName')], axis=1)
        df = pd.concat([df.drop('federalDistrictCode', axis=1),
                        encode_column_train(df, 'federalDistrictCode', 'federalDistrictCode')], axis=1)
        df = pd.concat([df.drop('code_professional_sphere', axis=1),
                        encode_column_train(df, 'code_professional_sphere', 'code_professional_sphere')], axis=1)

        premium_scaler = StandardScaler()
        df['additional_premium'] = premium_scaler.fit_transform(np.array(df['additional_premium']).reshape(-1, 1))
        experience_scaler = StandardScaler()
        df['required_experience'] = experience_scaler.fit_transform(np.array(df['required_experience']).reshape(-1, 1))

        mean_salary_max = df.loc[df['salary_max'] > 0, 'salary_max'].mean()
        df['salary_max'] = df['salary_max'].replace(0, mean_salary_max)
        df.loc[df['salary'] == 0, 'salary'] = df['salary_max']
        salary_scaler = StandardScaler()
        df['salary'] = salary_scaler.fit_transform(np.array(df['salary']).reshape(-1, 1))
        places_scaler = StandardScaler()
        df['work_places'] = places_scaler.fit_transform(np.array(df['work_places']).reshape(-1, 1))
        df = hash_col(df, 'code_profession', 100)
        df = df.dropna()
    else:
        df = fillna_with_mode(df)
        df = pd.concat([df.drop('academic_degree', axis=1), encode_column(df, 'academic_degree', 'academic_degree')],
                       axis=1)
        df = pd.concat(
            [df.drop('accommodation_type', axis=1), encode_column(df, 'accommodation_type', 'accommodation_type')],
            axis=1)
        df = pd.concat([df.drop('bonus_type', axis=1), encode_column(df, 'bonus_type', 'bonus_type')], axis=1)
        df = pd.concat([df.drop('measure_type', axis=1), encode_column(df, 'measure_type', 'measure_type')], axis=1)
        df = pd.concat([df.drop('busy_type', axis=1), encode_column(df, 'busy_type', 'busy_type')], axis=1)
        df = pd.concat([df.drop('education', axis=1), encode_column(df, 'education', 'education')], axis=1)
        df = pd.concat([df.drop('original_source_type', axis=1),
                        encode_column(df, 'original_source_type', 'original_source_type')], axis=1)
        df = pd.concat([df.drop('company_business_size', axis=1),
                        encode_column(df, 'company_business_size', 'company_business_size')], axis=1)
        df = pd.concat([df.drop('schedule_type', axis=1), encode_column(df, 'schedule_type', 'schedule_type')], axis=1)
        df = pd.concat([df.drop('social_protected_ids', axis=1),
                        encode_column(df, 'social_protected_ids', 'social_protected_ids')], axis=1)
        df = pd.concat([df.drop('source_type', axis=1), encode_column(df, 'source_type', 'source_type')], axis=1)
        df = pd.concat(
            [df.drop('state_region_code', axis=1), encode_column(df, 'state_region_code', 'state_region_code')], axis=1)
        df = pd.concat([df.drop('status', axis=1), encode_column(df, 'status', 'status')], axis=1)
        df = pd.concat([df.drop('transport_compensation', axis=1),
                        encode_column(df, 'transport_compensation', 'transport_compensation')], axis=1)
        df = pd.concat(
            [df.drop('vacancy_benefit_ids', axis=1), encode_column(df, 'vacancy_benefit_ids', 'vacancy_benefit_ids')],
            axis=1)
        df = pd.concat([df.drop('professionalSphereName', axis=1),
                        encode_column(df, 'professionalSphereName', 'professionalSphereName')], axis=1)
        df = pd.concat(
            [df.drop('federalDistrictCode', axis=1), encode_column(df, 'federalDistrictCode', 'federalDistrictCode')],
            axis=1)
        df = pd.concat([df.drop('code_professional_sphere', axis=1),
                        encode_column(df, 'code_professional_sphere', 'code_professional_sphere')], axis=1)
        df['additional_premium'] = premium_scaler.transform(np.array(df['additional_premium']).reshape(-1, 1))
        df['required_experience'] = experience_scaler.transform(np.array(df['required_experience']).reshape(-1, 1))
        df['work_places'] = places_scaler.transform(np.array(df['work_places']).reshape(-1, 1))
        df = hash_col(df, 'code_profession', 100)
        df = df.fillna(0)
    df = df.drop('vacancy_name', axis=1)
    if is_train or 'salary_min' in df.columns:
        df = df.drop('salary_min', axis=1)
        df = df.drop('salary_max', axis=1)

    return df




def regression(X_test):
    model = CatBoostRegressor()
    init_df = pd.read_csv('data/TRAIN_SAL.csv')
    # Load the model
    model.load_model('models/regression/regression.cbm')
    id = X_test['id']
    print(len(X_test))
    X_test = preprocess(X_test, is_train=False, train_df=init_df)
    X_test.to_csv('check.csv')
    y_pred = model.predict(X_test)
    y_pred_rubles = salary_scaler.inverse_transform(y_pred.reshape(-1,1))
    res = pd.DataFrame(columns=['id','task_type','salary'])
    res['id'] = id
    res['task_type'] = 'SAL'
    res['salary'] = y_pred_rubles
    return res

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='My CLI tool')
    parser.add_argument('-job', type=str, required=True, help='path to job test .csv file')
    parser.add_argument('-sal', type=str, required=True, help='path to sallary test .csv file')
    parser.add_argument('-sub', type=str, required=True, help='where to save submission .csv file')

    args = parser.parse_args()
    job_test = pd.read_csv(args.job)
    sal_test = pd.read_csv(args.sal)

    # Compute the first task
    #job_result = classificatoin(job_test)

    # Compute the second task
    sal_result = regression(sal_test)

    # Save the result
    #submission = pd.concat([job_result, sal_result], axis=0)
    #submission.to_csv(args.sub, index = False)


if __name__ == "__main__":
    main()