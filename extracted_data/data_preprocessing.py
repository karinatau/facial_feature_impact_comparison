import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial import distance

DATA_DIR_PATH = './extracted_data/data'
TESTS_AND_TRAIN_DIR_PATH='./extracted_data/occupations_vectors'
EXTRACTED_DATA = './extracted_data'

def data_extraction():
    print("data_extraction started")
    ## files with relevant data are Abilities, Skills, Knowledge, Work Activities, Work Styles, Work Values

    abilities = pd.read_excel("./extracted_data/ONET_db/Abilities.xlsx")
    skills = pd.read_excel("./extracted_data/ONET_db/Skills.xlsx")
    knowledge = pd.read_excel("./extracted_data/ONET_db//Knowledge.xlsx")
    work_Activities = pd.read_excel("./extracted_data/ONET_db/Work Activities.xlsx")
    work_styles = pd.read_excel("./extracted_data/ONET_db//Work Styles.xlsx")
    work_values = pd.read_excel("./extracted_data/ONET_db//Work Values.xlsx")

    # extract data - occupations as columns
    df_list=[abilities,skills,knowledge,work_Activities,work_styles]
    result_list =[]
    for df in df_list:
        titles_as_columns = pd.DataFrame(index=df['Element Name'].unique())
        for occupation in df['Title'].unique():
            temp=df[(df['Title']==occupation) & (df['Scale ID']=='IM')].loc[:,['Element Name','Data Value']]
            temp=temp.set_index(['Element Name'])
            temp.rename(columns = {'Data Value':occupation}, inplace = True)
            titles_as_columns = titles_as_columns.join(temp, how="outer")#למה join
        result_list.append(titles_as_columns)


    titles_as_columns = pd.DataFrame(index=work_values['Element Name'].unique())
    for occupation in work_values['Title'].unique():
        temp=work_values[(work_values['Title']==occupation) ].loc[:,['Element Name','Data Value']]
        temp=temp.set_index(['Element Name'])
        temp.rename(columns = {'Data Value':occupation}, inplace = True)
        titles_as_columns = titles_as_columns.join(temp, how="outer")
    result_list.append(titles_as_columns)
    result = pd.concat(result_list)

    result.drop(['Legislators'],axis=1,inplace=True)

    # add O*NET-SOC Code as number
    # round
    num_and_round_SOC_codes= lambda i: int(i[0:2]+i[3:7])
    vnum_and_round_SOC_codes= np.vectorize(num_and_round_SOC_codes)

    # not round
    num_SOC_codes= lambda i: float(i[0:2]+i[3:10])
    vnum_SOC_codes= np.vectorize(num_SOC_codes)

    result_with_SOC_code= result.copy()
    result_with_SOC_code.loc["round O*NET-SOC Code"] = vnum_and_round_SOC_codes(abilities['O*NET-SOC Code'].unique())
    result_with_SOC_code.loc["O*NET-SOC Code"] = vnum_SOC_codes(abilities['O*NET-SOC Code'].unique())

    result_with_SOC_code.to_csv(f'{EXTRACTED_DATA}/occupations_as_columns_with_SOC_CODE.csv')

    print("data_extraction finished successfully")


def pca(round_soc_family,proximity):
    print("pca started")
    vectors_with_SOC = pd.read_csv(f'{EXTRACTED_DATA}/occupations_as_columns_with_SOC_CODE.csv').set_index('Unnamed: 0').T
    
    soc_family_vectors = vectors_with_SOC[vectors_with_SOC["round O*NET-SOC Code"].isin(round_soc_family)]
    soc_family_vectors_without_soc = soc_family_vectors.drop(['O*NET-SOC Code','round O*NET-SOC Code'],axis=1,inplace=False)
    
    ### PCA with StandardScaler ###
    standard_vectors = StandardScaler().fit_transform(soc_family_vectors_without_soc)
    
    pca = PCA(n_components = len(soc_family_vectors_without_soc))
    principalComponents = pca.fit_transform(standard_vectors)

    principalComponents = pd.DataFrame(principalComponents,soc_family_vectors.index)

    # save vectors after PCA with SOC code
    principalComponents["O*NET-SOC Code"] = soc_family_vectors["O*NET-SOC Code"] 
    principalComponents["round O*NET-SOC Code"] = soc_family_vectors["round O*NET-SOC Code"]

    principalComponents.to_csv(f'{DATA_DIR_PATH}/principalComponents{proximity}.csv')
    print("pca finished successfully")

def class_and_vectors_alignment_conf(conf):
    if conf == 0: 
        return 0, [113031.0, 131199.0, 172051.0, 172141.0, 292099.0 ], 'train.csv', True, 'Familiar' 
    elif conf == 2.1:
        return 0, [113031.0, 131199.0, 172051.0, 172141.0, 292099.0 ], 'test2_1.csv', False, 'Familiar'
    elif conf == 3.1:
        return 1, [113031.0, 131199.0, 172051.0, 172141.0, 292099.0 ], 'test3_1.csv', False, 'Familiar'

def extract_class_names():
    class_names_file = open("./extracted_data/class_names.txt", "r") # opens the file in read mode and then 
    class_names = class_names_file.read().split() # puts the file into an array
    class_names.sort()
    class_names_file.close()
    return class_names
        

def class_and_vectors_alignment(conf, number_of_classes = 1050):
    # conf = 0 => train 
    # conf = 2 => test 2
    # conf = 2.1 => test 2 familiar face unfamiliar occupation
    # conf = 2.2 => test 2 inside family
    # conf = 2.3 => test 2 outside family
    # conf = 3.1 => test 3.1 unfamiliar face familiar occupation
    if conf == 2:
        create_test_two()
    elif conf == 2.2:
        create_test_two_inside_family()
    elif conf ==2.3:
        create_test_two_outside_family()
    else:
        other_tests(conf, number_of_classes)



def other_tests(conf, number_of_classes = 1050):
    # conf = 0 => train 
    # conf = 2.1 => test 2.1 familiar face unfamiliar occupation 
    # conf = 3.1 => test 3.1 unfamiliar face familiar occupation 

    i, relevant_occupation_vectors_soc_code, csv_name, create_min_and_median_csv, vector_source = class_and_vectors_alignment_conf(conf)

    min_and_median_result_path = f'{EXTRACTED_DATA}/occupation_familys_min_and_median.csv'

    # extract sorted class names 
    class_names = extract_class_names()

    # calculate median, and min distance within occupation family
    # create normal distribution df with correct standard deviation to each row (later we will add the occupation df with the normal distribution df)
    occupation_vectors = pd.read_csv( f'{DATA_DIR_PATH}/principalComponents{vector_source}.csv').drop(["Unnamed: 0"], axis = 1)
    #print(occupation_vectors.to_string())

    distance_df = pd.DataFrame()
    normal_distribution_df = pd.DataFrame()
    for round_SOC in relevant_occupation_vectors_soc_code:
        occupation_family = occupation_vectors[round(occupation_vectors["round O*NET-SOC Code"]) == round_SOC].drop(['round O*NET-SOC Code','O*NET-SOC Code'], axis=1)
        distance_matrix = distance.cdist(occupation_family, occupation_family, 'euclidean')
        distance_matrix = distance_matrix.reshape((9))
        distance_matrix = distance_matrix[distance_matrix != 0]

        distance_df = distance_df.append({"min":distance_matrix.min(), "median":np.median(distance_matrix), "round O*NET-SOC Code":round_SOC}, ignore_index=True)
        normal_distribution_df =pd.concat([normal_distribution_df,pd.DataFrame(np.random.normal(0,np.median(distance_matrix)*2/3, (int(number_of_classes/5),15)))])
        # normal_distribution_df = pd.concat([normal_distribution_df, pd.DataFrame(np.random.normal(0, 12, (int(number_of_classes / 5), 15)))])

    normal_distribution_df.set_axis(class_names[i*number_of_classes:(i+1)*number_of_classes], inplace=True)

    if create_min_and_median_csv:
        distance_df.to_csv(min_and_median_result_path)

    # extract the relevant occupation vectors
    relevant_occupation_vectors_df = occupation_vectors[occupation_vectors["round O*NET-SOC Code"].isin(relevant_occupation_vectors_soc_code)]
    # print(relevant_occupation_vectors_df.to_string())

    # calc distance matrix between vectors
    # total_distance_matrix = np.zeros((15,15))
    # for i in range(15):
    #     i_row = relevant_occupation_vectors_df.iloc[i, :-2]
    #     # print(i_row)
    #     for j in range(15):
    #         j_row = relevant_occupation_vectors_df.iloc[j, :-2]
    #         total_distance_matrix[i,j] = distance.euclidean(i_row, j_row)
    # print(total_distance_matrix)
    # total_dist_mean = total_distance_matrix.sum() / (15*15)
    # print(total_dist_mean)

    # row1 = relevant_occupation_vectors_df.iloc[0, :-2]
    # row15 = relevant_occupation_vectors_df.iloc[14, :-2]
    # tmp_dist = distance.euclidean(row1, row15)
    # print(tmp_dist)

    # Create df with the wanted number of each occupation vector
    relevant_occupation_vectors_df = pd.concat([relevant_occupation_vectors_df]*int(number_of_classes/len(relevant_occupation_vectors_df)),ignore_index=True)

    # Sort by O*NET-SOC Code In order to fit relevant_occupation_vectors_df to normal_distribution_df 
    relevant_occupation_vectors_df.sort_values(["O*NET-SOC Code"],inplace =True)
    # Adjust the structure of relevant_occupation_vectors_df to normal_distribution_df
    relevant_occupation_vectors_df.set_index(normal_distribution_df.index,inplace=True)
    soc_code_column = relevant_occupation_vectors_df['O*NET-SOC Code']
    round_soc_code_column = relevant_occupation_vectors_df['round O*NET-SOC Code']
    relevant_occupation_vectors_df.drop(['O*NET-SOC Code','round O*NET-SOC Code'],axis=1,inplace=True)
    relevant_occupation_vectors_df.columns = normal_distribution_df.columns

    # Add relevant_occupation_vectors_df and normal_distribution_df, the result is the wanted number_of_classes 
    # occupation vectors added with normal distribution vectors
    # number_of_classes_occupation_vectors_with_random_vectors = relevant_occupation_vectors_df
    number_of_classes_occupation_vectors_with_random_vectors = relevant_occupation_vectors_df + normal_distribution_df
    number_of_classes_occupation_vectors_with_random_vectors['O*NET-SOC Code'] = soc_code_column
    number_of_classes_occupation_vectors_with_random_vectors['round O*NET-SOC Code'] = round_soc_code_column

    if(conf == 3):
        sub_class_names_group = class_names[i*number_of_classes:(i+1)*number_of_classes] 
        new_sub_class_lst = [0 for k in range(number_of_classes)]
        for j in range (15):
            for k in range(14):
                new_sub_class_lst[((j + k + 1) * 70)%number_of_classes + (k * 5): ((j + k + 1) * 70)%number_of_classes + (k+1)*5] = sub_class_names_group[j * 70 + k*5:j * 70 + (k+1)*5]

        number_of_classes_occupation_vectors_with_random_vectors.set_axis(new_sub_class_lst,inplace=True)
        
        number_of_classes_occupation_vectors_with_random_vectors.sort_index(inplace = True)
        
    number_of_classes_occupation_vectors_with_random_vectors.to_csv(f'{TESTS_AND_TRAIN_DIR_PATH}/{csv_name}')

def create_test_two_outside_family():
    from numpy import random
    train_df = pd.read_csv(f'{TESTS_AND_TRAIN_DIR_PATH}/train.csv')
    classes = train_df['Unnamed: 0']
    train_df.drop(['Unnamed: 0'],axis =1, inplace=True)
    test_2_outside_family = train_df.copy()
    used_nums = []
    for i in range(4):
        count =0
        while (count<210):
            x = int(random.uniform(0, 1050))
            if ((x<210*i or x>=210*(i+1)) and x not in used_nums):
                used_nums.append(x)
                test_2_outside_family[count +210 * i:count +210 * i+1] = train_df[x:x+1]
                count+=1
    count = 0
    for i in range(1050):
        if i not in used_nums:
            test_2_outside_family[count +840:count +840+1] = train_df[i:i+1]
            count+=1
    test_2_outside_family.set_axis(classes, inplace=True)
    test_2_outside_family.to_csv(f'{TESTS_AND_TRAIN_DIR_PATH}/test2_3.csv') #2.3

def create_test_two_inside_family():
    train_df = pd.read_csv(f'{TESTS_AND_TRAIN_DIR_PATH}/train.csv')
    classes = train_df['Unnamed: 0']
    train_df.drop(['Unnamed: 0'],axis =1, inplace=True)
    test_2_inside_family = train_df.copy()
    for i in range(30):
        test_2_inside_family[210*int(i/6)+(i*35+(i%2+1)*70)%210:210*int(i/6)+(i*35+(i%2+1)*70)%210 +35 ] = train_df[i*35:(i+1)*35]
    test_2_inside_family.set_axis(classes, inplace=True)
    test_2_inside_family.to_csv(f'{TESTS_AND_TRAIN_DIR_PATH}/test2_2.csv') # 2.2

def create_test_two():
    train_df = pd.read_csv("./extracted_data/occupations_vectors/train.csv")
    train_df.drop(['Unnamed: 0'],axis =1, inplace=True)
    test_2 = train_df.copy()
    for j in range (15):
        for k in range(14):
            test_2[((j + k + 1) * 70)%1050 + (k * 5): ((j + k + 1) * 70)%1050 + (k+1)*5] = train_df[j * 70 + k*5:j * 70 + (k+1)*5]
    test_2
    test_2.to_csv("./extracted_data/occupations_vectors/test2.csv")

def data_preprocessing():
    # data_extraction()
    pca([113031.0, 131199.0, 172051.0, 172141.0, 292099.0 ],"Familiar")
    pca([119121.0, 131081.0, 191031.0, 292011.0, 518013.0],"Unfamiliar")
    # class_and_vectors_alignment(3.1)

if __name__ == '__main__':
    data_preprocessing()
