#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def data_to_spherical_to_euclidean(df_input):
    # First Create Spherical Coordinates From DataFrame
    x1 = np.cos(df_input.latitude) * np.cos(df_input.longitude)
    x2 = np.cos(df_input.latitude) * np.sin(df_input.longitude)
    x3 = np.sin(df_input.latitude)
    # Write to Data-Frame
    coordinates = {"x1": x1, "x2": x2, "x3": x3}
    coordinates = pd.DataFrame(data=coordinates, index=df_input.index)

    # Compute Extrinsic (Euclidean) Mean and Project onto Sphere
    x_bar = np.mean(coordinates, axis=0)
    x_bar = x_bar / np.linalg.norm(x_bar)

    # Map to Euclidean Coordinates about the projected extrinsic mean
    def Log_Sphere(p):
        # Compute dot product between x and p
        x_dot_p = np.matmul(x_bar, p)
        # Compute Angle Between x and p
        x_p_ang = np.arccos(x_dot_p)
        # Spherical "projection" factor
        x_p_fact = x_p_ang / (np.sqrt(1 - (x_dot_p ** 2)))
        # Compute Coordinate on Tangent Space
        tangent_space_val = (p - x_bar * x_dot_p) * x_p_fact
        # Return Ouput
        return tangent_space_val

    # Return Result
    result = [Log_Sphere(row) for row in coordinates.values]
    return pd.DataFrame(result, index=df_input.index)


# In[ ]:


def feature_map(df_input):
    ret_vec = data_to_spherical_to_euclidean(df_input)
    df_enriched = pd.concat([df_input, ret_vec], axis=1)
    # Reset Index from 1
    df_enriched = df_enriched.reset_index(drop=True)
    return df_enriched


# In[ ]:


def prepare_columntransformer(cl):
    min_max = [
        "housing_median_age",
        "total_rooms",
        "population",
        "households",
        "median_income",
    ]
    standard = ["longitude", "latitude", "x1", "x2", "x3"]
    standard_idx = [i for i, obj in enumerate(cl) if obj in standard]
    min_max_idx = [i for i, obj in enumerate(cl) if obj in min_max]
    ct = ColumnTransformer(
        [
            ("Min-Max", MinMaxScaler(), min_max_idx),
            ("zero-one", StandardScaler(), standard_idx),
        ],
        remainder="passthrough",
    )

    return ct


def prepare_data(X_input_data, manual, test_size=test_size_ratio):
    # Initialize
    X = X_input_data
    # Fit Oddities
    y = np.array(X["median_house_value"])
    y = y * (10 ** (-5))
    X = X.drop(["median_house_value"], axis=1)
    # Prepare Training/Testing Splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=2000
    ) 
    
    return X_train, y_train, X_test, y_test


# #### Implied Hyper-parameter(s)

# In[ ]:


# Remove 1 part because of the identity map!
# N_parts = max(1,N_parts)

# Ensure that there are at-least as many random feature maps generated as there are parts to the partition!
# N_Features_Search_Space_Dimension = max(N_parts,N_Features_Search_Space_Dimension)

# Set number of input variable(s)
# Input_dimension_set = X_train.shape[1]



# Read Data
data_path = raw_data_path_folder+"housing.csv"
X_raw = pd.read_csv(data_path)
# Remember Column Names
X_colnames = X_raw.columns

# Add Euclidean Coordinates
X = feature_map(X_raw)
# Add Dummy Variables (Categorical)
X = pd.get_dummies(X)
# Remove total "bedrooms" since contains nan and can be inferred from total number of rooms (see Kaggle discussion)
X = X.drop(columns='total_bedrooms')
# Define Training / Testing Sets
X_train, y_train, X_test, y_test = prepare_data(X,False)


# Fit Standardizer
min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(X_train)
# Standardize Training Set
X_train = min_max_scaler.transform(X_train)
X_train = pd.DataFrame(X_train)
# Standardize Testing Set with Training Values
X_test = min_max_scaler.transform(X_test)
X_test = pd.DataFrame(X_test)