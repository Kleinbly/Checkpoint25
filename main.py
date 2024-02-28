import streamlit as st

# title
st.title('Financial Inclusion in Africa')

# Prediction model
# Step1 : Importing the packages
import pandas as pd
import sklearn
# Supervised learning
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Label Encoders
encoder0 = sklearn.preprocessing.LabelEncoder()
encoder1 = sklearn.preprocessing.LabelEncoder()
encoder2 = sklearn.preprocessing.LabelEncoder()
encoder3 = sklearn.preprocessing.LabelEncoder()
encoder4 = sklearn.preprocessing.LabelEncoder()
encoder5 = sklearn.preprocessing.LabelEncoder()
encoder6 = sklearn.preprocessing.LabelEncoder()
encoder7 = sklearn.preprocessing.LabelEncoder()
encoder8 = sklearn.preprocessing.LabelEncoder()

def trainingClassifier():
    # Step2: Loading the dataset
    dataset = pd.read_csv('Financial_inclusion_dataset.csv')

    # Step3 : Encoding categorical values
    # Encoding the labels in column "REGION"
    # Encoding the labels in column "REGION"

    # Create a list of all the columns to encode
    encode_features_list = dataset.columns.tolist()
    encode_features_list.remove('household_size')
    encode_features_list.remove('age_of_respondent')

    dataset['country'] = encoder0.fit_transform(dataset['country'])
    dataset['year'] = encoder1.fit_transform(dataset['year'])
    dataset['location_type'] = encoder2.fit_transform(dataset['location_type'])
    dataset['cellphone_access'] = encoder3.fit_transform(dataset['cellphone_access'])
    dataset['gender_of_respondent'] = encoder4.fit_transform(dataset['gender_of_respondent'])
    dataset['relationship_with_head'] = encoder5.fit_transform(dataset['relationship_with_head'])
    dataset['marital_status'] = encoder6.fit_transform(dataset['marital_status'])
    dataset['education_level'] = encoder7.fit_transform(dataset['education_level'])
    dataset['job_type'] = encoder8.fit_transform(dataset['job_type'])

    # Step6: Supervised Learning
    # Features extraction
    x = dataset.drop(['uniqueid', 'bank_account'], axis=1)

    y = dataset['bank_account']

    # Splitting data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=10)

    # Applying tree algorithm
    tree_model = tree.DecisionTreeClassifier()
    tree_model.fit(x_train, y_train)  # Fitting our model
    y_pred = tree_model.predict(x_test)  # Evaluating our model
    model_accuracyScore = str(round(accuracy_score(y_test, y_pred), 4))

    return model_accuracyScore, tree_model


# inputs
country = st.selectbox('Sélectionner un pays de votre choix :', ['Kenya', 'Rwanda', 'Tanzania', 'Uganda'])
year = st.selectbox('Sélectionner une année :', [2018, 2016, 2017])
age_of_respondent = st.slider('Quel âge avez-vous? :', 0, 25, 100)
location_type = st.radio('Sélectionner un type de lieu :', ['Rural', 'Urban'])
cellphone_access = st.radio('Avez-vous accès à un téléphone portable :', ['Yes', 'No'])
gender_of_respondent = st.radio('Quel est votre genre :', ['Female', 'Male'])
household_size = st.slider('Entrez le nombre de personnes avec qui vous vivez dans une seule maison :', 0, 0, 30)
relationship_with_head = st.selectbox('Quel est votre relation avec le chef de famille :',
                                      ['Spouse', 'Head of Household', 'Other relative', 'Child', 'Parent',
                                       'Other non-relatives'])
marital_status = st.selectbox('Quel est votre statut marital :',
                              ['Married/Living together', 'Widowed', 'Single/Never Married', 'Divorced/Seperated',
                               'Dont know'])
education_level = st.selectbox("Quel est votre niveeau d'éducation :",
                               ['Secondary education', 'No formal education', 'Vocational/Specialised training',
                                'Primary education', 'Tertiary education', 'Other/Dont know/RTA'])
job_type = st.selectbox("Quel type d'emploi exercé :",
                        ['Self employed', 'Government Dependent', 'Formally employed Private', 'Informally employed',
                         'Formally employed Government', 'Farming and Fishing', 'Remittance Dependent', 'Other Income',
                         'Dont Know/Refuse to answer', 'No Income'])

if st.button("Predict", type="primary"):
    model_accuracy, model = trainingClassifier()

    # Encoding inputs
    country_encoded = encoder0.transform(pd.DataFrame([[country]], columns=['country']))
    year_encoded = encoder1.transform(pd.DataFrame([[year]], columns=['year']))
    location_type_encoded = encoder2.transform(pd.DataFrame([[location_type]], columns=['location_type']))
    cellphone_access_encoded = encoder3.transform(pd.DataFrame([[cellphone_access]], columns=['cellphone_access']))
    gender_of_respondent_encoded = encoder4.transform(pd.DataFrame([[gender_of_respondent]], columns=['gender_of_respondent']))
    relationship_with_head_encoded = encoder5.transform(pd.DataFrame([[relationship_with_head]], columns=['relationship_with_head']))
    marital_status_encoded = encoder6.transform(pd.DataFrame([[marital_status]], columns=['marital_status']))
    education_level_encoded = encoder7.transform(pd.DataFrame([[education_level]], columns=['education_level']))
    job_type_encoded = encoder8.transform(pd.DataFrame([[job_type]], columns=['job_type']))

    # Add the inputs for prediction
    new_row = {
        'country' : country_encoded,
        'year' : year_encoded,
        'location_type' : location_type_encoded,
        'cellphone_access' : cellphone_access_encoded,
        'household_size' : household_size,
        'age_of_respondent' : age_of_respondent,
        'gender_of_respondent' : gender_of_respondent_encoded,
        'relationship_with_head' : relationship_with_head_encoded,
        'marital_status' : marital_status_encoded,
        'education_level' : education_level_encoded,
        'job_type' : job_type_encoded
    }

    # Add the new row
    df = pd.DataFrame(new_row)

    # Result
    output = model.predict(df)

    # Display accuracy of the model
    st.info('The accuracy of the model is : ' + model_accuracy, icon="ℹ️")
    # Display the result
    st.success('This person is likely to have a bank account.' if output[0] == 1 else 'This person is likely to NOT have a bank account.',
               icon="ℹ️")
